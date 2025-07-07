import numpy as np

# def detect_para(bbox_dict):
#     alpha1 = 0.2
#     alpha2 = 0.7
#     beta1 = 0.4
#     data = bbox_dict
#     word_crops = list(data.keys())
#     for i in word_crops:
#         data[i]["x1"], data[i]["y1"], data[i]["x2"], data[i]["y2"] = data[i]["bbox"]
#         data[i]["xc"] = (data[i]["x1"] + data[i]["x2"]) / 2
#         data[i]["yc"] = (data[i]["y1"] + data[i]["y2"]) / 2
#         data[i]["w"] = data[i]["x2"] - data[i]["x1"]
#         data[i]["h"] = data[i]["y2"] - data[i]["y1"]

#     patch_info = {}
#     while word_crops:
#         img_name = word_crops[0].split("_")[0]
#         word_crop_collection = [
#             word_crop for word_crop in word_crops if word_crop.startswith(img_name)
#         ]
#         centroids = {}
#         lines = []
#         img_word_crops = word_crop_collection.copy()
#         para = []
#         while img_word_crops:
#             clusters = []
#             para_words_group = [
#                 img_word_crops[0],
#             ]
#             added = [
#                 img_word_crops[0],
#             ]
#             img_word_crops.remove(img_word_crops[0])
#             ## determining the paragraph
#             while added:
#                 word_crop = added.pop()
#                 for i in range(len(img_word_crops)):
#                     word_crop_ = img_word_crops[i]
#                     if (
#                         abs(data[word_crop_]["yc"] - data[word_crop]["yc"])
#                         < data[word_crop]["h"] * alpha1
#                     ):
#                         if data[word_crop]["xc"] > data[word_crop_]["xc"]:
#                             if (data[word_crop]["x1"] - data[word_crop_]["x2"]) < data[
#                                 word_crop
#                             ]["h"] * alpha2:
#                                 para_words_group.append(word_crop_)
#                                 added.append(word_crop_)
#                         else:
#                             if (data[word_crop_]["x1"] - data[word_crop]["x2"]) < data[
#                                 word_crop
#                             ]["h"] * alpha2:
#                                 para_words_group.append(word_crop_)
#                                 added.append(word_crop_)
#                     else:
#                         if data[word_crop]["yc"] > data[word_crop_]["yc"]:
#                             if (data[word_crop]["y1"] - data[word_crop_]["y2"]) < data[
#                                 word_crop
#                             ]["h"] * beta1 and (
#                                 (
#                                     (data[word_crop_]["x1"] < data[word_crop]["x2"])
#                                     and (data[word_crop_]["x1"] > data[word_crop]["x1"])
#                                 )
#                                 or (
#                                     (data[word_crop_]["x2"] < data[word_crop]["x2"])
#                                     and (data[word_crop_]["x2"] > data[word_crop]["x1"])
#                                 )
#                                 or (
#                                     (data[word_crop]["x1"] > data[word_crop_]["x1"])
#                                     and (data[word_crop]["x2"] < data[word_crop_]["x2"])
#                                 )
#                             ):
#                                 para_words_group.append(word_crop_)
#                                 added.append(word_crop_)
#                         else:
#                             if (data[word_crop_]["y1"] - data[word_crop]["y2"]) < data[
#                                 word_crop
#                             ]["h"] * beta1 and (
#                                 (
#                                     (data[word_crop_]["x1"] < data[word_crop]["x2"])
#                                     and (data[word_crop_]["x1"] > data[word_crop]["x1"])
#                                 )
#                                 or (
#                                     (data[word_crop_]["x2"] < data[word_crop]["x2"])
#                                     and (data[word_crop_]["x2"] > data[word_crop]["x1"])
#                                 )
#                                 or (
#                                     (data[word_crop]["x1"] > data[word_crop_]["x1"])
#                                     and (data[word_crop]["x2"] < data[word_crop_]["x2"])
#                                 )
#                             ):
#                                 para_words_group.append(word_crop_)
#                                 added.append(word_crop_)
#                 img_word_crops = [p for p in img_word_crops if p not in para_words_group]
#             ## processing for the line
#             while para_words_group:
#                 line_words_group = [
#                     para_words_group[0],
#                 ]
#                 added = [
#                     para_words_group[0],
#                 ]
#                 para_words_group.remove(para_words_group[0])
#                 ## determining the line
#                 while added:
#                     word_crop = added.pop()
#                     for i in range(len(para_words_group)):
#                         word_crop_ = para_words_group[i]
#                         if (
#                             abs(data[word_crop_]["yc"] - data[word_crop]["yc"])
#                             < data[word_crop]["h"] * alpha1
#                         ):
#                             if data[word_crop]["xc"] > data[word_crop_]["xc"]:
#                                 if (data[word_crop]["x1"] - data[word_crop_]["x2"]) < data[
#                                     word_crop
#                                 ]["h"] * alpha2:
#                                     line_words_group.append(word_crop_)
#                                     added.append(word_crop_)
#                             else:
#                                 if (data[word_crop_]["x1"] - data[word_crop]["x2"]) < data[
#                                     word_crop
#                                 ]["h"] * alpha2:
#                                     line_words_group.append(word_crop_)
#                                     added.append(word_crop_)
#                     para_words_group = [
#                         p for p in para_words_group if p not in line_words_group
#                     ]
#                 xc = [data[word_crop]["xc"] for word_crop in line_words_group]
#                 idxs = np.argsort(xc)
#                 patch_cluster_ = [line_words_group[i] for i in idxs]
#                 line_words_group = patch_cluster_
#                 x1 = [data[word_crop]["x1"] for word_crop in line_words_group]
#                 x2 = [data[word_crop]["x2"] for word_crop in line_words_group]
#                 y1 = [data[word_crop]["y1"] for word_crop in line_words_group]
#                 y2 = [data[word_crop]["y2"] for word_crop in line_words_group]
#                 txt_line = [data[word_crop]["txt"] for word_crop in line_words_group]
#                 txt = " ".join(txt_line)
#                 x = [x1[0]]
#                 y1_ = [y1[0]]
#                 y2_ = [y2[0]]
#                 l = [len(txt_l) for txt_l in txt_line]
#                 for i in range(1, len(x1)):
#                     x.append((x1[i] + x2[i - 1]) / 2)
#                     y1_.append((y1[i] + y1[i - 1]) / 2)
#                     y2_.append((y2[i] + y2[i - 1]) / 2)
#                 x.append(x2[-1])
#                 y1_.append(y1[-1])
#                 y2_.append(y2[-1])
#                 line_info = {
#                     "x": x,
#                     "y1": y1_,
#                     "y2": y2_,
#                     "l": l,
#                     "txt": txt,
#                     "word_crops": line_words_group,
#                 }
#                 clusters.append(line_info)
#             y_ = [clusters[i]["y1"][0] for i in range(len(clusters))]
#             idxs = np.argsort(y_)
#             clusters_ = [clusters[i] for i in idxs]
#             txt = [clusters[i]["txt"] for i in idxs]
#             l = [len(t) for t in txt]
#             txt = " ".join(txt)
#             para_info = {"lines": clusters_, "l": l, "txt": txt}
#             para.append(para_info)

#         for word_crop in word_crop_collection:
#             word_crops.remove(word_crop)
#         return "\n".join([para[i]["txt"] for i in range(len(para))])


def detect_para(recognized_texts):
    """
    Sort words into lines based on horizontal overlap of bounding boxes.
    
    Args:
        recognized_texts (dict): A dictionary with recognized texts as keys and bounding boxes as values.
                                 Each bounding box is a list of points [x1, y1, x2, y2].
    
    Returns:
        list: A list of lists where each sublist contains words sorted by x-coordinate for a single line.
    """
    def calculate_overlap(bbox1, bbox2):
        """Calculate the vertical overlap between two bounding boxes."""
        # Extract bounding box coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        height = min(y2_1 - y1_1, y2_2 - y1_2)
        return overlap / height if height > 0 else 0

    # Convert recognized_texts dictionary to a list of tuples for processing
    items = list(recognized_texts.items())
    lines = []

    while items:
        current_image, current_data = items.pop(0)
        current_text, current_bbox = current_data['txt'], current_data['bbox']
        current_line = [(current_text, current_bbox)]

        remaining_items = []
        for image, data in items:
            text, bbox = data['txt'], data['bbox']
            if calculate_overlap(current_bbox, bbox) > 0.4:
                current_line.append((text, bbox))
            else:
                remaining_items.append((image, data))

        items = remaining_items
        lines.append(current_line)

    # Sort words within each line based on x1 (horizontal position)
    sorted_lines = [
        [text for text, bbox in sorted(line, key=lambda x: x[1][0])] for line in lines
    ]
    return sorted_lines


