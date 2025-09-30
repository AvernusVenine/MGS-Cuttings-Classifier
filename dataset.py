class CuttingLabel:
    SHALE = 'shale'
    FELSIC = 'felsic'

CLASS_MAP = {
    1 : CuttingLabel.SHALE,
    0 : CuttingLabel.FELSIC,
}

import xml.etree.ElementTree as ET
import os
import glob
from pathlib import Path
import shutil

def validate_xml_annotation(xml_path):
    """
    Validate a single XML annotation file for invalid bounding boxes

    Args:
        xml_path: Path to the XML annotation file

    Returns:
        dict: Validation results with details about any issues
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract basic info
        filename = root.find('filename').text if root.find('filename') is not None else "unknown"
        image_width = int(root.find('size/width').text) if root.find('size/width') is not None else None
        image_height = int(root.find('size/height').text) if root.find('size/height') is not None else None

        # Find all objects
        objects = root.findall('object')

        invalid_boxes = []
        valid_boxes = []
        all_boxes = []

        for i, obj in enumerate(objects):
            name = obj.find('name').text if obj.find('name') is not None else "unknown"
            bndbox = obj.find('bndbox')

            if bndbox is not None:
                try:
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    box = [xmin, ymin, xmax, ymax]
                    all_boxes.append(box)

                    # Calculate width and height
                    width = xmax - xmin
                    height = ymax - ymin

                    # Check for invalid boxes
                    issues = []
                    if width <= 0:
                        issues.append(f"Invalid width: {width}")
                    if height <= 0:
                        issues.append(f"Invalid height: {height}")
                    if xmin < 0:
                        issues.append(f"xmin < 0: {xmin}")
                    if ymin < 0:
                        issues.append(f"ymin < 0: {ymin}")
                    if image_width and xmax > image_width:
                        issues.append(f"xmax > image_width: {xmax} > {image_width}")
                    if image_height and ymax > image_height:
                        issues.append(f"ymax > image_height: {ymax} > {image_height}")

                    box_info = {
                        'object_index': i,
                        'class_name': name,
                        'bbox': box,
                        'width': width,
                        'height': height,
                        'issues': issues
                    }

                    if issues:
                        invalid_boxes.append(box_info)
                    else:
                        valid_boxes.append(box_info)

                except (ValueError, AttributeError) as e:
                    invalid_boxes.append({
                        'object_index': i,
                        'class_name': name,
                        'bbox': None,
                        'error': f"Error parsing bbox: {e}",
                        'issues': [f"Parse error: {e}"]
                    })

        result = {
            'xml_path': xml_path,
            'filename': filename,
            'image_size': (image_width, image_height),
            'total_objects': len(objects),
            'valid_boxes': valid_boxes,
            'invalid_boxes': invalid_boxes,
            'is_valid': len(invalid_boxes) == 0,
            'all_boxes': all_boxes
        }

        return result

    except ET.ParseError as e:
        return {
            'xml_path': xml_path,
            'filename': "unknown",
            'error': f"XML parse error: {e}",
            'is_valid': False,
            'invalid_boxes': [{'error': str(e), 'issues': ['XML parse error']}]
        }
    except Exception as e:
        return {
            'xml_path': xml_path,
            'filename': "unknown",
            'error': f"Unexpected error: {e}",
            'is_valid': False,
            'invalid_boxes': [{'error': str(e), 'issues': ['Unexpected error']}]
        }


def validate_all_xml_annotations(xml_dir, pattern="*.xml"):
    """
    Validate all XML annotation files in a directory

    Args:
        xml_dir: Directory containing XML files
        pattern: File pattern to match (default: "*.xml")

    Returns:
        tuple: (valid_files, invalid_files) with detailed results
    """
    xml_files = glob.glob(os.path.join(xml_dir, pattern))

    print(f"Found {len(xml_files)} XML files in {xml_dir}")
    print("Validating annotations...")

    valid_files = []
    invalid_files = []

    for xml_path in xml_files:
        result = validate_xml_annotation(xml_path)

        if result['is_valid']:
            valid_files.append(result)
            print(f"✅ {os.path.basename(xml_path)}")
        else:
            invalid_files.append(result)
            print(f"❌ {os.path.basename(xml_path)}")

            # Print detailed error info
            if 'error' in result:
                print(f"   Error: {result['error']}")
            else:
                for box_info in result['invalid_boxes']:
                    print(f"   Object {box_info.get('object_index', '?')}: {', '.join(box_info['issues'])}")
                    if box_info.get('bbox'):
                        print(f"     Bbox: {box_info['bbox']}")

    print(f"\nValidation complete:")
    print(f"Valid files: {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")

    return valid_files, invalid_files


def find_specific_box(xml_dir, target_box, tolerance=0.01):
    """
    Find the XML file containing a specific problematic bounding box

    Args:
        xml_dir: Directory containing XML files
        target_box: The problematic box coordinates [xmin, ymin, xmax, ymax]
        tolerance: Tolerance for floating point comparison

    Returns:
        List of matches with file info
    """
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    matches = []

    print(f"Searching {len(xml_files)} XML files for box: {target_box}")

    for i, xml_path in enumerate(xml_files):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = root.find('filename').text if root.find('filename') is not None else "unknown"
            objects = root.findall('object')

            for obj_idx, obj in enumerate(objects):
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    try:
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)

                        current_box = [xmin, ymin, xmax, ymax]

                        # Check if this box matches the target (within tolerance)
                        if all(abs(current_box[j] - target_box[j]) <= tolerance for j in range(4)):
                            class_name = obj.find('name').text if obj.find('name') is not None else "unknown"

                            match_info = {
                                'xml_file': xml_path,
                                'image_file': filename,
                                'object_index': obj_idx,
                                'class_name': class_name,
                                'bbox': current_box,
                                'width': xmax - xmin,
                                'height': ymax - ymin
                            }
                            matches.append(match_info)

                            print(f"🎯 FOUND MATCH!")
                            print(f"   File: {xml_path}")
                            print(f"   Image: {filename}")
                            print(f"   Object {obj_idx} ({class_name})")
                            print(f"   Box: {current_box}")
                            print(f"   Width: {xmax - xmin}, Height: {ymax - ymin}")

                    except (ValueError, AttributeError) as e:
                        continue

        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            continue

    if not matches:
        print("❌ No exact matches found. Searching for boxes with zero width/height...")

        # Search for any zero-width or zero-height boxes
        for xml_path in xml_files:
            try:
                result = validate_xml_annotation(xml_path)
                if not result['is_valid'] and result.get('invalid_boxes'):
                    for invalid_box in result['invalid_boxes']:
                        if invalid_box.get('bbox'):
                            box = invalid_box['bbox']
                            width = box[2] - box[0]
                            height = box[3] - box[1]
                            if width <= 0 or height <= 0:
                                print(f"🔍 Found invalid box in {xml_path}:")
                                print(f"   Box: {box}, Width: {width}, Height: {height}")
                                matches.append({
                                    'xml_file': xml_path,
                                    'image_file': result.get('filename', 'unknown'),
                                    'bbox': box,
                                    'width': width,
                                    'height': height,
                                    'issues': invalid_box.get('issues', [])
                                })
            except:
                continue

    return matches
