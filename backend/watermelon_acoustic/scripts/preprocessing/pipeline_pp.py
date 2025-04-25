#!/usr/bin/env python
import os
import sys
import shutil
import functions

# Import your existing modules
from scripts.preprocessing.wav_conversion import convert_qilin_file_formats_to_wav
from scripts.preprocessing.standardization import standardize_wav_files
from scripts.preprocessing.noise_reduction import apply_noise_reduction
from scripts.preprocessing.normalization import normalize_audio_files
from scripts.preprocessing.feature_extraction import extract_features
from scripts.preprocessing.feature_generation import generate_features  # renamed module for clarity
from scripts.preprocessing.feature_selection import select_features
from scripts.preprocessing.set_splitting import pre_split_holdouts


class PipelineStage:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def run(self, context):
        functions.green_print(f"\nRunning stage: {self.name}")
        return self.func(context)


class Pipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, context):
        for stage in self.stages:
            context = stage.run(context)
        return context


# Stage functions – each accepts a context dictionary and returns it (possibly updated).

def stage_reformat_dataset(context):
    base_dir = context['base_dir']
    use_qilin = context.get('USE_QILIN', True)
    qilin_dataset_dir = os.path.join(base_dir, "../../input", "qilin_dataset", "19_datasets")
    qilin_preprocess_dir = os.path.join(base_dir, "../../input", "wav_qilin")
    context['qilin_preprocess_dir'] = qilin_preprocess_dir

    if use_qilin:
        print("[FC] Starting Qilin dataset preprocessing...")
        convert_qilin_file_formats_to_wav(qilin_dataset_dir, qilin_preprocess_dir)
    else:
        print("[FC] Skipping Qilin dataset preprocessing.")
    return context


def stage_standardize(context):
    base_dir = context['base_dir']
    use_qilin = context.get('USE_QILIN', True)
    use_lab = context.get('USE_LAB', False)

    if use_qilin:
        qilin_preprocess_dir = context.get('qilin_preprocess_dir')
        qilin_standard_dir = os.path.join(base_dir, "../../intermediate", "standard_qilin")
        context['qilin_standard_dir'] = qilin_standard_dir
        print("[SW] Standardizing Qilin dataset...")
        standardize_wav_files(qilin_preprocess_dir, qilin_standard_dir)

    if use_lab:
        lab_dataset_dir = os.path.join(base_dir, "../../input", "wav_lab")
        lab_standard_dir = os.path.join(base_dir, "../../intermediate", "standard_lab")
        context['lab_standard_dir'] = lab_standard_dir
        print("[SW] Standardizing Lab dataset...")
        standardize_wav_files(lab_dataset_dir, lab_standard_dir)

    return context


def stage_combine_datasets(context):
    base_dir = context['base_dir']
    use_qilin = context.get('USE_QILIN', True)
    use_lab = context.get('USE_LAB', False)

    if use_qilin and use_lab:
        print(f"[CD] Using both Qilin and lab datasets")
        qilin_standard_dir = context.get('qilin_standard_dir')
        lab_standard_dir = context.get('lab_standard_dir')
        combined_standard_dir = os.path.join(base_dir, "../../intermediate", "combined_standard")
        functions.combine_folders(qilin_standard_dir, lab_standard_dir, combined_standard_dir)
        context['combined_standard_dir'] = combined_standard_dir
    elif use_qilin:
        print(f"[CD] Using only Qilin dataset")
        context['combined_standard_dir'] = context.get('qilin_standard_dir')
    elif use_lab:
        print(f"[CD] Using only lab dataset")
        context['combined_standard_dir'] = context.get('lab_standard_dir')
    else:
        print("[CD] No dataset selected for training. Please select at least one.")
        sys.exit(1)
    return context


def stage_noise_reduction(context):
    base_dir = context['base_dir']
    combined_standard_dir = context.get('combined_standard_dir')
    noise_reduction_dir = os.path.join(base_dir, "../../intermediate", "noise_reduction")
    functions.clear_output_directory(noise_reduction_dir)
    apply_noise_reduction(combined_standard_dir, noise_reduction_dir)
    context['noise_reduction_dir'] = noise_reduction_dir
    return context


def stage_normalization(context):
    base_dir = context['base_dir']
    noise_reduction_dir = context.get('noise_reduction_dir')
    print("[NM] Normalizing audio files...")
    normalize_audio_files(noise_reduction_dir, noise_reduction_dir)
    return context


def stage_feature_extraction(context):
    base_dir = context['base_dir']
    noise_reduction_dir = context.get('noise_reduction_dir')
    feature_extraction_base_dir = os.path.join(base_dir, "../../intermediate", "feature_extraction")
    functions.clear_output_directory(feature_extraction_base_dir)

    # Process each noise reduction technique subfolder
    for technique in os.listdir(noise_reduction_dir):
        technique_path = os.path.join(noise_reduction_dir, technique)
        if os.path.isdir(technique_path):
            output_feat_dir = os.path.join(feature_extraction_base_dir, technique)
            functions.clear_output_directory(output_feat_dir)
            extract_features(technique_path, output_feat_dir)

    context['feature_extraction_base_dir'] = feature_extraction_base_dir
    return context


def stage_duplicate_feature_extraction(context):
    """
    Duplicate the feature_extraction_base_dir into two folders:
      one for further feature generation/selection ("with_fs")
      and one to leave raw ("without_fs").
    """
    base_dir = context['base_dir']
    feat_base = context.get('feature_extraction_base_dir')
    with_fs = os.path.join(base_dir, "../../intermediate", "feature_extraction_with_fs")
    without_fs = os.path.join(base_dir, "../../intermediate", "feature_extraction_without_fs")
    functions.clear_output_directory(with_fs)
    functions.clear_output_directory(without_fs)

    # Copy entire feature_extraction_base_dir into both new folders
    print(f"[DP] Duplicating {os.path.relpath(feat_base, base_dir)} into {os.path.relpath(with_fs, base_dir)}")
    shutil.copytree(feat_base, with_fs, dirs_exist_ok=True)
    print(f"[DP] Duplicating {os.path.relpath(feat_base, base_dir)} into {os.path.relpath(without_fs, base_dir)}")
    shutil.copytree(feat_base, without_fs, dirs_exist_ok=True)

    context['feature_extraction_with_fs'] = with_fs
    context['feature_extraction_without_fs'] = without_fs
    print(f"[DP] Duplicated feature extraction folder into:\n  With FS: {with_fs}\n  Without FS: {without_fs}")
    return context


def stage_feature_generation_fs(context):
    """
    Run feature generation (e.g., Yeo-Johnson, robust scaling, poly features)
    only on the "with_fs" branch.
    """
    print("[FG] Running feature generation (transformation) on the with_fs branch...")
    generate_features(context['feature_extraction_with_fs'])
    return context


def stage_feature_selection_fs(context, k=50):
    """
    Run feature selection (e.g., top 50 via SelectKBest) on the "with_fs" branch.
    """
    print("[FS] Running feature selection on the with_fs branch...")
    select_features(context['feature_extraction_with_fs'], k)
    return context


def stage_set_splitting_fs(context):
    """
    Split the "with_fs" branch into training and hold-out sets.
    """
    base_dir = context['base_dir']
    feat_with_fs = context['feature_extraction_with_fs']
    print("[SS] Splitting with_fs branch into training and hold-out sets...")
    pre_split_holdouts(feat_with_fs, holdout_ratio=context.get('TEST_SPLIT_RATIO', 0.15))
    return context


def stage_set_splitting_without_fs(context):
    """
    Split the "without_fs" branch into training and hold-out sets.
    """
    base_dir = context['base_dir']
    feat_without_fs = context['feature_extraction_without_fs']
    print("[SS] Splitting without_fs branch into training and hold-out sets...")
    pre_split_holdouts(feat_without_fs, holdout_ratio=context.get('TEST_SPLIT_RATIO', 0.15))
    return context


def stage_completion(context):
    print("[PP] Preprocessing complete. Ready for model training.")
    # input("[PP] Press Enter to exit...")
    return context


def main(USE_QILIN, USE_LAB, TEST_SPLIT_RATIO=0.15, K=50):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Set initial context
    context = {
        'base_dir': base_dir,
        'USE_QILIN': USE_QILIN,
        'USE_LAB': USE_LAB,
        'TEST_SPLIT_RATIO': TEST_SPLIT_RATIO
    }

    # Define pipeline stages in order.
    # After feature extraction, we duplicate the folder and then process each branch.
    stages = [
        PipelineStage("Reformat Dataset", stage_reformat_dataset),
        PipelineStage("Standardize Wav Files", stage_standardize),
        PipelineStage("Combine Datasets", stage_combine_datasets),
        PipelineStage("Noise Reduction", stage_noise_reduction),
        PipelineStage("Normalization", stage_normalization),
        PipelineStage("Feature Extraction", stage_feature_extraction),
        PipelineStage("Duplicate Feature Extraction", stage_duplicate_feature_extraction),
        # Process branch that will have feature generation/selection:
        PipelineStage("Feature Generation (with FS)", stage_feature_generation_fs),
        PipelineStage("Feature Selection (with FS)", stage_feature_selection_fs),
        PipelineStage("Set Splitting (with FS)", stage_set_splitting_fs),
        # Process branch that skips generation/selection:
        PipelineStage("Set Splitting (without FS)", stage_set_splitting_without_fs),
        PipelineStage("Completion", stage_completion)
    ]

    pipeline = Pipeline(stages)
    pipeline.run(context)


if __name__ == "__main__":
    main(True, False)
