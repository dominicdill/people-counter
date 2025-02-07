# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2



# def show_points(coords, labels, ax, marker_size=100):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

# def show_point(coords, label, ax, marker_size=100):
#     if label==0:
#       color='red'
#     else:
#       color='green'
#     ax.scatter([coords[0]], [coords[1]], color=color, marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
     
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

# def show_boxes(boxes, ax):
#     for box in boxes:
#       show_box(box,ax)

# def show_yolo_box(yolo_box, ax, height, width):
#     class_id = yolo_box.pop(0)
#     edgecolor = 'green' if class_id == 0 else 'red'
#     xc, yc, w, h = yolo_box[0]*width, yolo_box[1]*height, yolo_box[2]*width, yolo_box[3]*height
#     x0 = xc - w/2
#     y0 = yc - h/2
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))

# def show_yolo_boxes(yolo_boxes, ax, height, width):
#     for yolo_box in yolo_boxes:
#       show_yolo_box(yolo_box, ax, height, width)

# def get_yolo_boxes_from_annotation_file(annotation_file):
#     boxes = []
#     with open(annotation_file, 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         box = line.split(' ')
#         box = [float(x) for x in box]
#         boxes.append(box)
#     return boxes



# def on_key(event):
#     """
#     Only advance to the next image when 'n' or 'enter/return' is pressed.
#     Quit entirely when 'q' is pressed.
#     All other key events, including mouse clicks, are ignored.
#     """
#     if event.key in ['n', 'enter', 'return']:
#         # Close current figure to show next image
#         plt.close()
#     elif event.key == 'q':
#         # Close all figures and exit the program
#         plt.close('all')
#         sys.exit(0)
#     else:
#         # Ignore any other key presses
#         pass

# def display_images(folder):
#     # Loop through sorted files in the folder (only processing .jpg images)
#     for filename in sorted(os.listdir(folder)):
#         if not filename.lower().endswith('.jpg'):
#             continue

#         # Load and convert the image from BGR to RGB
#         img_path = os.path.join(folder, filename)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue  # Skip if the image cannot be loaded
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Construct the annotation file path
#         annotations_folder = folder.replace('images', 'labels')
#         annotation_filename = filename.replace('.jpg', '.txt')
#         annotation_path = os.path.join(annotations_folder, annotation_filename)
#         print("Loading annotations from:", annotation_path)
#         boxes = get_yolo_boxes_from_annotation_file(annotation_path)

#         # Create a new figure to display the image and annotations
#         fig, ax = plt.subplots()
#         ax.imshow(img)
#         height, width, _ = img.shape
#         show_yolo_boxes(boxes, ax, height, width)

#         # Connect the key press event to our callback
#         fig.canvas.mpl_connect('key_press_event', on_key)
#         print(f"Displaying {filename}. Press 'n' or 'enter' to go to the next image, or 'q' to quit.")

#         # Show the image (this call blocks until plt.close() is called)
#         plt.get_current_fig_manager().window.showMaximized()
#         plt.show()

# # Replace with the path to your images folder
# display_images('dataset/images/train')



import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# =========================
# Helper functions for display
# =========================

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_point(coords, label, ax, marker_size=100):
    color = 'red' if label==0 else 'green'
    ax.scatter([coords[0]], [coords[1]], color=color, marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
     
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def show_boxes(boxes, ax):
    for box in boxes:
      show_box(box, ax)

# (If you still need the original yolo box functions, you may adjust them so they do not modify the box list)
def show_yolo_box(yolo_box, ax, height, width):
    # Do not pop the class id so that the box is not modified!
    class_id = yolo_box[0]
    xc = yolo_box[1] * width
    yc = yolo_box[2] * height
    w = yolo_box[3] * width
    h = yolo_box[4] * height
    x0 = xc - w/2
    y0 = yc - h/2
    edgecolor = 'green' if class_id == 0 else 'red'
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))

def show_yolo_boxes(yolo_boxes, ax, height, width):
    for yolo_box in yolo_boxes:
      show_yolo_box(yolo_box, ax, height, width)

def get_yolo_boxes_from_annotation_file(annotation_file):
    boxes = []
    if not os.path.exists(annotation_file):
        return boxes  # return empty list if no file
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # Each line: "class_id x_center y_center width height"
        box = line.strip().split(' ')
        if len(box) != 5:
            continue  # skip invalid lines
        box = [float(x) for x in box]
        boxes.append(box)
    return boxes

# =========================
# The interactive BoxEditor class
# =========================

class BoxEditor:
    def __init__(self, ax, fig, boxes, annotation_path, img_width, img_height):
        """
        Parameters:
          ax: matplotlib axes where the image is shown.
          fig: the current figure.
          boxes: a list of boxes in YOLO format [class_id, x_center_norm, y_center_norm, width_norm, height_norm].
          annotation_path: path to the annotation file (to be updated on save).
          img_width, img_height: image dimensions in pixels.
        """
        self.ax = ax
        self.fig = fig
        self.boxes = boxes  # list of boxes in YOLO format
        self.annotation_path = annotation_path
        self.img_width = img_width
        self.img_height = img_height

        # For adding new boxes (left click)
        self.temp_point = None  # stores the first corner (x, y) when left clicking
        self.temp_marker = None  # a temporary marker for the first click

        # A list to hold the drawn rectangle patches for each box
        self.patches = []
        for box in self.boxes:
            patch = self.draw_box(box)
            self.patches.append(patch)
        
        # Connect the mouse click event
        self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

    def draw_box(self, yolo_box):
        """Draw a rectangle for a box in YOLO format and return the patch."""
        class_id = yolo_box[0]
        xc = yolo_box[1] * self.img_width
        yc = yolo_box[2] * self.img_height
        w = yolo_box[3] * self.img_width
        h = yolo_box[4] * self.img_height
        x0 = xc - w/2
        y0 = yc - h/2
        edgecolor = 'green' if class_id == 0 else 'red'
        patch = plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2)
        self.ax.add_patch(patch)
        self.fig.canvas.draw_idle()
        return patch

    def on_click(self, event):
        """Callback for mouse button press events."""
        if event.inaxes != self.ax:
            return  # ignore clicks outside the axes

        if event.button == 1:  # Left click: add box
            self.handle_left_click(event)
        elif event.button == 3:  # Right click: remove box
            self.handle_right_click(event)

    def handle_left_click(self, event):
        if self.temp_point is None:
            # Save the first corner and mark it
            self.temp_point = (event.xdata, event.ydata)
            self.temp_marker = self.ax.scatter(event.xdata, event.ydata, color='yellow', marker='o')
            self.fig.canvas.draw_idle()
        else:
            # Second left click: define the opposite corner
            x1, y1 = self.temp_point
            x2, y2 = event.xdata, event.ydata
            # Determine top-left and bottom-right coordinates
            x0 = min(x1, x2)
            y0 = min(y1, y2)
            x1_ = max(x1, x2)
            y1_ = max(y1, y2)
            w = x1_ - x0
            h = y1_ - y0
            # Ignore very small boxes
            if w < 1 or h < 1:
                self.clear_temp()
                return
            # Convert the pixel coordinates into YOLO format (normalized)
            xc = x0 + w/2
            yc = y0 + h/2
            x_center_norm = xc / self.img_width
            y_center_norm = yc / self.img_height
            w_norm = w / self.img_width
            h_norm = h / self.img_height
            new_box = [0, x_center_norm, y_center_norm, w_norm, h_norm]
            # Append the new box and draw it
            self.boxes.append(new_box)
            patch = self.draw_box(new_box)
            self.patches.append(patch)
            self.clear_temp()

    def clear_temp(self):
        """Clear the temporary first click marker."""
        if self.temp_marker:
            self.temp_marker.remove()
            self.temp_marker = None
        self.temp_point = None
        self.fig.canvas.draw_idle()

    def handle_right_click(self, event):
        """Remove a box if the click is inside one."""
        click_x, click_y = event.xdata, event.ydata
        idx_to_remove = None
        # Iterate through boxes and their patches
        for idx, box in enumerate(self.boxes):
            xc = box[1] * self.img_width
            yc = box[2] * self.img_height
            w = box[3] * self.img_width
            h = box[4] * self.img_height
            x0 = xc - w/2
            y0 = yc - h/2
            if x0 <= click_x <= x0 + w and y0 <= click_y <= y0 + h:
                idx_to_remove = idx
                break
        if idx_to_remove is not None:
            # Remove the box and its patch
            self.boxes.pop(idx_to_remove)
            patch = self.patches.pop(idx_to_remove)
            patch.remove()
            self.fig.canvas.draw_idle()

    def disconnect(self):
        """Disconnect the mouse click event."""
        self.fig.canvas.mpl_disconnect(self.cid)

    def save_annotations(self):
        """Save the current list of boxes (in YOLO format) to the annotation file."""
        with open(self.annotation_path, 'w') as f:
            for box in self.boxes:
                line = ' '.join(str(x) for x in box)
                f.write(line + '\n')

# =========================
# Key press callback for the editor
# =========================

def on_key_editor(event, editor):
    """
    When 'n' or 'enter' is pressed, save the annotation file and close the figure.
    When 'q' is pressed, save and quit the program.
    """
    if event.key in ['n', 'enter', 'return']:
        editor.save_annotations()
        editor.disconnect()
        plt.close()
    elif event.key == 'q':
        editor.save_annotations()
        editor.disconnect()
        plt.close('all')
        sys.exit(0)
    else:
        # Ignore any other keys
        pass

# =========================
# Main display function (modified to include editing)
# =========================

def display_images(folder):
    """
    For each .jpg image in the folder, display it along with its current annotations.
    Then allow the user to add boxes with left-clicks and remove boxes with right-clicks.
    Press 'n' or 'enter' to save changes and move to the next image,
    or press 'q' to quit the program.
    """
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith('.jpg'):
            continue

        # Load image (convert from BGR to RGB)
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip if image cannot be loaded
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Construct the annotation file path.
        annotations_folder = folder.replace('images', 'labels')
        annotation_filename = filename.replace('.jpg', '.txt')
        annotation_path = os.path.join(annotations_folder, annotation_filename)
        print("Loading annotations from:", annotation_path)
        boxes = get_yolo_boxes_from_annotation_file(annotation_path)

        # Create figure and axis; display the image.
        fig, ax = plt.subplots()
        ax.imshow(img)
        height, width, _ = img.shape

        # (Optional) Draw the original boxes with your existing function:
        # show_yolo_boxes(boxes, ax, height, width)

        # Create a BoxEditor to enable interactive editing.
        editor = BoxEditor(ax, fig, boxes, annotation_path, width, height)

        # Connect the key press event; we pass the editor to the callback via a lambda.
        fig.canvas.mpl_connect('key_press_event', lambda event: on_key_editor(event, editor))
        print(f"Displaying {filename}.")
        print("  Left-click twice to add a box; right-click inside a box to remove it.")
        print("  Press 'n' or 'enter' to save and move to the next image, or 'q' to quit.")

        # Maximize window if possible and show the figure (this call blocks until the figure is closed)
        mng = plt.get_current_fig_manager()
        try:
            mng.window.showMaximized()
        except AttributeError:
            pass  # Some backends may not support this

        plt.show()

# =========================
# Run the viewer/editor
# =========================

# Replace with the path to your images folder.
display_images('dataset/images/train')
