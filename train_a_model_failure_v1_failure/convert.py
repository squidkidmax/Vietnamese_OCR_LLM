import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
import argparse
from tqdm import tqdm

def make_image_file(coord_groups, output_path: str, line_width, dpi):
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    for group in coord_groups:
        plt.plot(group[:, 0], group[:, 1], linewidth=line_width, c='black')

    if dpi is None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

def convert(ink_files, out_img_dir, out_label_path, line_width=2, dpi=None):
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
        
    annotations = pd.DataFrame(columns=['id', 'label'])
    total_files = len(ink_files)
    print(f"Processing {total_files} files for {out_img_dir}")
    processed_files = 0
    generated_images = 0

    for idx, inkml_file in enumerate(tqdm(ink_files, desc="Processing files")):
        if not os.path.exists(inkml_file):
            print(f"Skipping missing file: {inkml_file}")
            continue
        try:
            tree = ET.parse(inkml_file)
            root = tree.getroot()
            trace_groups = root.findall('traceGroup')
            if not trace_groups:
                print(f"No traceGroup found in {inkml_file}")
                continue

            for sample in trace_groups:
                sample_id = os.path.splitext(os.path.basename(inkml_file))[0] + '_' + sample.get('id', 'NO_ID')
                sample_label = sample.find('.//Tg_Truth')
                if sample_label is None or not sample_label.text:
                    print(f"No valid Tg_Truth in {inkml_file} for sample {sample_id}")
                    continue
                sample_label = sample_label.text.strip()
                new_row = pd.DataFrame([{'id': sample_id, 'label': sample_label}])
                annotations = pd.concat([annotations, new_row], ignore_index=True)

                coord_groups = []
                for trace_tag in sample.findall('trace'):
                    coord_group = []
                    trace_text = trace_tag.text.strip() if trace_tag.text else ''
                    if not trace_text:
                        print(f"Empty trace in {inkml_file} for sample {sample_id}")
                        continue
                    for coord_text in trace_text.split(','):
                        coord_text = coord_text.strip()
                        if not coord_text:
                            continue
                        coords_str = [c for c in coord_text.split(' ') if c]
                        try:
                            coords = np.array([int(c) for c in coords_str])
                            if len(coords) == 2:
                                coord_group.append(coords)
                            else:
                                print(f"Skipping invalid coords in {inkml_file}: {coords} (from '{coord_text}') - length != 2")
                        except ValueError as e:
                            print(f"Skipping non-integer coords in {inkml_file}: '{coord_text}' - error: {e}")
                    if coord_group:
                        coord_groups.append(np.stack(coord_group))
                    else:
                        print(f"Skipping empty coord_group in {inkml_file} for trace: {trace_text}")
                if coord_groups:
                    make_image_file(coord_groups, os.path.join(out_img_dir, sample_id) + '.png', line_width, dpi)
                    generated_images += 1
                else:
                    print(f"Skipping image generation for {sample_id} in {inkml_file} - no valid coord_groups")
            processed_files += 1
        except Exception as e:
            print(f"Error processing file {inkml_file}: {e}")
            continue
    print(f"Completed processing {processed_files}/{total_files} files. Generated {generated_images} images.")
    print(f"Saving annotations to {out_label_path} with {len(annotations)} entries")
    annotations.to_csv(out_label_path, sep='\t', index=False)
    if annotations.empty:
        print(f"Warning: No annotations generated for {out_label_path}")

def convert_label_only(ink_files, out_label_path):
    annotations = pd.DataFrame(columns=['id', 'label'])
    total_files = len(ink_files)
    print(f"Processing {total_files} files for labels to {out_label_path}")
    processed_files = 0

    for idx, inkml_file in enumerate(tqdm(ink_files, desc="Processing files for labels")):
        if not os.path.exists(inkml_file):
            print(f"Skipping missing file: {inkml_file}")
            continue
        try:
            tree = ET.parse(inkml_file)
            root = tree.getroot()
            trace_groups = root.findall('traceGroup')
            if not trace_groups:
                print(f"No traceGroup found in {inkml_file}")
                continue

            for sample in trace_groups:
                sample_id = os.path.splitext(os.path.basename(inkml_file))[0] + '_' + sample.get('id', 'NO_ID')
                sample_label = sample.find('.//Tg_Truth')
                if sample_label is None or not sample_label.text:
                    print(f"No valid Tg_Truth in {inkml_file} for sample {sample_id}")
                    continue
                sample_label = sample_label.text.strip()
                new_row = pd.DataFrame([{'id': sample_id, 'label': sample_label}])
                annotations = pd.concat([annotations, new_row], ignore_index=True)
            processed_files += 1
        except Exception as e:
            print(f"Error processing file {inkml_file} for labels: {e}")
            continue
    print(f"Completed processing {processed_files}/{total_files} files. Generated {len(annotations)} label entries.")
    annotations.to_csv(out_label_path, sep='\t', index=False)
    if annotations.empty:
        print(f"Warning: No annotations generated for {out_label_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, dest='level', choices=['word', 'line', 'paragraph'])
    parser.add_argument('-w', '--line_width', type=float, dest='line_width', default=2)
    parser.add_argument('-dpi', '--dpi', type=int, dest='dpi', default=None)
    parser.add_argument('--label_only', action='store_true', dest='label_only')

    args = parser.parse_args()
    data_dir = './data'
    level = args.level

    inkml_dir = os.path.join(data_dir, f'InkData_{level}')
    out_label_train = os.path.join(data_dir, f'train_{level}.csv')
    out_label_validation = os.path.join(data_dir, f'validation_{level}.csv')
    out_label_test = os.path.join(data_dir, f'test_{level}.csv')
    out_label_all = os.path.join(data_dir, f'all_{level}.csv')

    icfhr_datasplit_dir = os.path.join(data_dir, 'VNOnDB_ICFHR2018_dataSplit')
    train_set = os.path.join(icfhr_datasplit_dir, 'train_set.txt')
    val_set = os.path.join(icfhr_datasplit_dir, 'validation_set.txt')
    test_set = os.path.join(icfhr_datasplit_dir, 'test_set.txt')

    with open(train_set) as f:
        train_ink_files = [os.path.join(inkml_dir, line.rstrip()) for line in f]
    with open(val_set) as f:
        val_ink_files = [os.path.join(inkml_dir, line.rstrip()) for line in f]
    with open(test_set) as f:
        test_ink_files = [os.path.join(inkml_dir, line.rstrip()) for line in f]

    print('number train_ink_files:', len(train_ink_files))
    print('number val_ink_files:', len(val_ink_files))
    print('number test_ink_files:', len(test_ink_files))

    if args.label_only:
        convert_label_only(train_ink_files, out_label_train)
        convert_label_only(val_ink_files, out_label_validation)
        convert_label_only(test_ink_files, out_label_test)
    else:
        out_img_train = os.path.join(data_dir, f'train_{level}')
        out_img_validation = os.path.join(data_dir, f'validation_{level}')
        out_img_test = os.path.join(data_dir, f'test_{level}')

        if not os.path.exists(out_img_train):
            os.mkdir(out_img_train)
        if not os.path.exists(out_img_validation):
            os.mkdir(out_img_validation)
        if not os.path.exists(out_img_test):
            os.mkdir(out_img_test)

        line_width = args.line_width
        dpi = args.dpi

        convert(train_ink_files, out_img_train, out_label_train, line_width, dpi)
        convert(val_ink_files, out_img_validation, out_label_validation, line_width, dpi)
        convert(test_ink_files, out_img_test, out_label_test, line_width, dpi)

    convert_label_only(train_ink_files + val_ink_files + test_ink_files, out_label_all)