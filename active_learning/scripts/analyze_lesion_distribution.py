#!/usr/bin/env python3
"""
Analyze VinDr-CXR lesion distribution and categorize into concentrated/dispersed.

This script helps understand lesion characteristics and provides statistics
for experimental design.
"""

import csv
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

# Add active_learning/ (parent of scripts/) to path for flat imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.public.vindr_cxr import CONCENTRATED_LESIONS, DISPERSED_LESIONS, LESION_MAPPING


def analyze_lesion_distribution(data_root='/team/team_pxi/pxi-dataset/cxr/public/vinbig'):
    """Analyze lesion distribution and categorize."""
    data_root = Path(data_root)
    annotation_file_train = data_root / 'old' / 'annotations_train.csv'
    annotation_file_test = data_root / 'old' / 'annotations_test.csv'
    
    def get_stats(annotation_file, split_name):
        """Get statistics for a split."""
        lesion_image_counts = defaultdict(set)  # lesion -> set of image_ids
        lesion_bbox_counts = Counter()  # lesion -> total bboxes
        image_bbox_counts = defaultdict(int)  # image_id -> number of bboxes
        
        with open(annotation_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['image_id']
                class_name = row['class_name']
                
                # Skip "No finding"
                if class_name == 'No finding':
                    continue
                
                # Check if bbox exists
                if row.get('x_min') and row.get('y_min') and row.get('x_max') and row.get('y_max'):
                    try:
                        float(row['x_min'])  # Valid bbox
                        lesion_image_counts[class_name].add(image_id)
                        lesion_bbox_counts[class_name] += 1
                        image_bbox_counts[image_id] += 1
                    except:
                        pass
        
        return lesion_image_counts, lesion_bbox_counts, image_bbox_counts
    
    # Analyze train and test
    train_img_counts, train_bbox_counts, train_img_bbox_counts = get_stats(annotation_file_train, 'train')
    test_img_counts, test_bbox_counts, test_img_bbox_counts = get_stats(annotation_file_test, 'test')
    
    # Combine all lesions
    all_lesions = set(train_img_counts.keys()) | set(test_img_counts.keys())
    
    # Map VinDr classes to our lesion names
    class_to_lesion = {}
    for lesion, classes in LESION_MAPPING.items():
        for cls in classes:
            class_to_lesion[cls] = lesion
    
    # Categorize lesions
    concentrated_classes = set()
    for lesion, classes in CONCENTRATED_LESIONS.items():
        concentrated_classes.update(classes)
    
    dispersed_classes = set()
    for lesion, classes in DISPERSED_LESIONS.items():
        dispersed_classes.update(classes)
    
    print("=" * 80)
    print("📊 VinDr-CXR Lesion Distribution Analysis")
    print("=" * 80)
    
    print("\n🔵 CONCENTRATED LESIONS (집중된 병변)")
    print("-" * 80)
    print(f"{'Lesion':<25} {'Train Img':<12} {'Train Bbox':<12} {'Test Img':<12} {'Test Bbox':<12} {'Avg Bbox/Img':<12}")
    print("-" * 80)
    
    concentrated_stats = []
    for cls in sorted(concentrated_classes):
        if cls in all_lesions:
            train_imgs = len(train_img_counts[cls])
            train_bbox = train_bbox_counts[cls]
            test_imgs = len(test_img_counts[cls])
            test_bbox = test_bbox_counts[cls]
            avg_bbox = (train_bbox + test_bbox) / max(1, train_imgs + test_imgs)
            
            lesion_name = class_to_lesion.get(cls, cls)
            print(f"{lesion_name:<25} {train_imgs:<12} {train_bbox:<12} {test_imgs:<12} {test_bbox:<12} {avg_bbox:<12.2f}")
            concentrated_stats.append((lesion_name, train_imgs, test_imgs, train_bbox, test_bbox, avg_bbox))
    
    print("\n🟢 DISPERSED LESIONS (분산된 병변)")
    print("-" * 80)
    print(f"{'Lesion':<25} {'Train Img':<12} {'Train Bbox':<12} {'Test Img':<12} {'Test Bbox':<12} {'Avg Bbox/Img':<12}")
    print("-" * 80)
    
    dispersed_stats = []
    for cls in sorted(dispersed_classes):
        if cls in all_lesions:
            train_imgs = len(train_img_counts[cls])
            train_bbox = train_bbox_counts[cls]
            test_imgs = len(test_img_counts[cls])
            test_bbox = test_bbox_counts[cls]
            avg_bbox = (train_bbox + test_bbox) / max(1, train_imgs + test_imgs)
            
            lesion_name = class_to_lesion.get(cls, cls)
            print(f"{lesion_name:<25} {train_imgs:<12} {train_bbox:<12} {test_imgs:<12} {test_bbox:<12} {avg_bbox:<12.2f}")
            dispersed_stats.append((lesion_name, train_imgs, test_imgs, train_bbox, test_bbox, avg_bbox))
    
    # Calculate average bboxes per image for each category
    print("\n" + "=" * 80)
    print("📈 Summary Statistics")
    print("=" * 80)
    
    # Concentrated: calculate from images
    concentrated_total_imgs = sum(train_imgs + test_imgs for _, train_imgs, test_imgs, _, _, _ in concentrated_stats)
    concentrated_total_bbox = sum(train_bbox + test_bbox for _, _, _, train_bbox, test_bbox, _ in concentrated_stats)
    concentrated_avg = concentrated_total_bbox / max(1, concentrated_total_imgs)
    
    # Dispersed: calculate from images
    dispersed_total_imgs = sum(train_imgs + test_imgs for _, train_imgs, test_imgs, _, _, _ in dispersed_stats)
    dispersed_total_bbox = sum(train_bbox + test_bbox for _, _, _, train_bbox, test_bbox, _ in dispersed_stats)
    dispersed_avg = dispersed_total_bbox / max(1, dispersed_total_imgs)
    
    print(f"\n🔵 Concentrated Lesions:")
    print(f"   Total images: {concentrated_total_imgs}")
    print(f"   Total bboxes: {concentrated_total_bbox}")
    print(f"   Avg bboxes per image: {concentrated_avg:.2f}")
    
    print(f"\n🟢 Dispersed Lesions:")
    print(f"   Total images: {dispersed_total_imgs}")
    print(f"   Total bboxes: {dispersed_total_bbox}")
    print(f"   Avg bboxes per image: {dispersed_avg:.2f}")
    
    # Analyze multi-bbox images
    print("\n" + "=" * 80)
    print("📊 Multi-Bbox Analysis (Images with multiple bboxes)")
    print("=" * 80)
    
    def analyze_multi_bbox(image_bbox_counts, split_name):
        """Analyze images with multiple bboxes."""
        multi_bbox_images = {img: count for img, count in image_bbox_counts.items() if count > 1}
        single_bbox_images = {img: count for img, count in image_bbox_counts.items() if count == 1}
        
        print(f"\n{split_name.upper()}:")
        print(f"  Total images with bboxes: {len(image_bbox_counts)}")
        print(f"  Single bbox images: {len(single_bbox_images)} ({100*len(single_bbox_images)/max(1,len(image_bbox_counts)):.1f}%)")
        print(f"  Multi bbox images: {len(multi_bbox_images)} ({100*len(multi_bbox_images)/max(1,len(image_bbox_counts)):.1f}%)")
        
        if multi_bbox_images:
            bbox_dist = Counter(multi_bbox_images.values())
            print(f"  Multi-bbox distribution:")
            for count, num_imgs in sorted(bbox_dist.items()):
                print(f"    {count} bboxes: {num_imgs} images")
        
        return len(multi_bbox_images), len(single_bbox_images)
    
    train_multi, train_single = analyze_multi_bbox(train_img_bbox_counts, 'Train')
    test_multi, test_single = analyze_multi_bbox(test_img_bbox_counts, 'Test')
    
    print("\n" + "=" * 80)
    print("✅ Recommended Experimental Setup")
    print("=" * 80)
    print("\n🔵 Concentrated Lesions (for experiment):")
    print("   --target_lesion calcification")
    print("   --target_lesion aortic_enlargement")
    print("   --target_lesion pneumothorax")
    print("   --target_lesion consolidation")
    print("   --target_lesion pleural_effusion")
    print("   --target_lesion cardiomegaly")
    
    print("\n🟢 Dispersed Lesions (for experiment):")
    print("   --target_lesion nodule")
    print("   --target_lesion infiltration")
    print("   --target_lesion lung_opacity")
    
    print("\n💡 Tip: You can use multiple lesions:")
    print("   --target_lesion calcification aortic_enlargement  # Multiple concentrated")
    print("   --target_lesion nodule infiltration  # Multiple dispersed")


if __name__ == '__main__':
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else '/team/team_pxi/pxi-dataset/cxr/public/vinbig'
    analyze_lesion_distribution(data_root)
