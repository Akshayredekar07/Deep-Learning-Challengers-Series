### **OpenCV Image Processing: Functions, Arguments, and Examples**

---

### **1. Installing and Importing OpenCV**
- Install OpenCV using `pip`:
  ```python
  %pip install opencv-python
  ```
- Import necessary libraries:
  ```python
  import cv2
  import numpy as np
  ```

---

### **2. `cv2.imread()` - Reading an Image**
#### **Function:** 
```python
cv2.imread(filename, flags=cv2.IMREAD_COLOR)
```
#### **Arguments:**
- `filename (str)`: Path to the image file.
- `flags (int, optional)`: 
  - `cv2.IMREAD_COLOR (1)`: Loads a color image (default).
  - `cv2.IMREAD_GRAYSCALE (0)`: Loads a grayscale image.
  - `cv2.IMREAD_UNCHANGED (-1)`: Loads an image with an alpha channel (transparency).

#### **Example:**
```python
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Gray Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **3. `cv2.imshow()` - Displaying an Image**
#### **Function:**
```python
cv2.imshow(window_name, image)
```
#### **Arguments:**
- `window_name (str)`: Name of the window.
- `image (numpy.ndarray)`: Image to be displayed.

#### **Example:**
```python
img = cv2.imread('image.png')
cv2.imshow('Displayed Image', img)
cv2.waitKey(0)  # Wait indefinitely for a key press
cv2.destroyAllWindows()
```

---

### **4. `cv2.waitKey()` - Pause Execution**
#### **Function:**
```python
cv2.waitKey(delay)
```
#### **Arguments:**
- `delay (int)`: Time in milliseconds to wait for a key press. `0` means indefinite wait.

#### **Example:**
```python
cv2.waitKey(2000)  # Wait for 2 seconds
```

---

### **5. `cv2.destroyAllWindows()` - Closing Windows**
#### **Function:**
```python
cv2.destroyAllWindows() # Close all OpenCV windows
```

---

### **6. `cv2.resize()` - Resizing an Image**
#### **Function:**
```python
cv2.resize(src, dsize, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
```
#### **Arguments:**
- `src (numpy.ndarray)`: Input image.
- `dsize (tuple)`: New size `(width, height)`.
- `fx, fy (float, optional)`: Scaling factors.
- `interpolation (int, optional)`: Resampling method (default is `cv2.INTER_LINEAR`).

#### **Example:**
```python
img = cv2.imread('image.png')
resized_img = cv2.resize(img, (300, 500))  # Resize to 300x500 pixels
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **7. `cv2.imwrite()` - Saving an Image**
#### **Function:**
```python
cv2.imwrite(filename, image)
```
#### **Arguments:**
- `filename (str)`: Name of the output image file.
- `image (numpy.ndarray)`: Image to be saved.

#### **Example:**
```python
cv2.imwrite('resized_image.png', resized_img)
```

---

### **8. `cv2.cvtColor()` - Converting Color Spaces**
#### **Function:**
```python
cv2.cvtColor(src, code)
```
#### **Arguments:**
- `src (numpy.ndarray)`: Input image.
- `code (int)`: Conversion code (e.g., `cv2.COLOR_BGR2RGB`).

#### **Example:**
```python
img = cv2.imread('image.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('RGB Image', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **9. `cv2.split()` & `cv2.merge()` - Splitting & Merging Channels**
#### **Functions:**
```python
b, g, r = cv2.split(image)
merged = cv2.merge([b, g, r])
```
#### **Arguments:**
- `image (numpy.ndarray)`: Input image.
- `b, g, r (numpy.ndarray)`: Individual color channels.
- `cv2.merge([b, g, r])`: Merges the channels back.

#### **Example:**
```python
img = cv2.imread('image.png')
b, g, r = cv2.split(img)
merged_img = cv2.merge([b, g, r])
cv2.imshow('Merged Image', merged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **10. `np.hstack()` & `np.vstack()` - Image Stacking**
#### **Functions:**
```python
horizontal = np.hstack((img1, img2))
vertical = np.vstack((img1, img2))
```
#### **Arguments:**
- `img1, img2 (numpy.ndarray)`: Images of the same size.

#### **Example:**
```python
img = cv2.imread('image.png')
resized = cv2.resize(img, (300, 500))

# Stack images horizontally
horizontal_stack = np.hstack((resized, resized))

# Stack images vertically
vertical_stack = np.vstack((horizontal_stack, horizontal_stack))

cv2.imshow('Stacked Image', vertical_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **11. Image Slideshow using `os.listdir()`**
#### **Function:**
```python
os.listdir(directory)
```
#### **Example:**
```python
import os

path = "D:\\Images"
image_list = os.listdir(path)

for img_name in image_list:
    img = cv2.imread(os.path.join(path, img_name))
    if img is not None:
        resized = cv2.resize(img, (500, 700))
        cv2.imshow('Slideshow', resized)
        cv2.waitKey(2000)  # Show each image for 2 seconds
        cv2.destroyAllWindows()
```

---

### **Summary**
| Function | Purpose | Key Arguments |
|----------|---------|---------------|
| `cv2.imread()` | Reads an image | `filename`, `flags` |
| `cv2.imshow()` | Displays an image | `window_name`, `image` |
| `cv2.waitKey()` | Pauses execution | `delay` |
| `cv2.destroyAllWindows()` | Closes all windows | N/A |
| `cv2.resize()` | Resizes an image | `src`, `dsize`, `fx`, `fy`, `interpolation` |
| `cv2.imwrite()` | Saves an image | `filename`, `image` |
| `cv2.cvtColor()` | Converts color spaces | `src`, `code` |
| `cv2.split()` | Splits image channels | `image` |
| `cv2.merge()` | Merges image channels | `[b, g, r]` |
| `np.hstack()` | Stacks images horizontally | `img1, img2` |
| `np.vstack()` | Stacks images vertically | `img1, img2` |
| `os.listdir()` | Lists files in a directory | `directory` |

---

### **Complete Syntax of `cv2.putText()`**  

```python
cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
```

### **Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `img` | Input image where text is placed | `img = cv2.imread("image.jpg")` |
| `text` | The string to be written | `"Hello OpenCV"` |
| `org` | Bottom-left corner of text `(x, y)` | `(50, 100)` |
| `fontFace` | Font style from OpenCV predefined fonts | `cv2.FONT_HERSHEY_SIMPLEX` |
| `fontScale` | Size multiplier for text | `1.5` (1.5x default size) |
| `color` | Text color in **BGR** format `(Blue, Green, Red)` | `(0, 255, 0)` (green) |
| `thickness` | Thickness of the text (integer) | `2` |
| `lineType` | Type of text line (smoothness) | `cv2.LINE_AA` |
| `bottomLeftOrigin` | `True` starts from bottom-left, `False` from top-left | `False` (default) |

---

### **Example Code: Writing Text on an Image**
```python
import cv2
import numpy as np

# Create a blank image
img = np.zeros((500, 800, 3), dtype=np.uint8)

# Add text to the image
cv2.putText(img, "Hello OpenCV!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
            2, (255, 0, 0), 3, cv2.LINE_AA)

# Display the image
cv2.imshow("Image with Text", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This code creates a black image and writes `"Hello OpenCV!"` in **blue (255,0,0)**, using **FONT_HERSHEY_SIMPLEX** with a font size of `2`, a thickness of `3`, and anti-aliased smooth edges (`cv2.LINE_AA`).

---

Here's the requested content in the same structured notes format:  

---

### **OpenCV: Drawing Bounding Boxes, Lines, and Shapes**  

---

### **1. `cv2.line()` - Drawing a Line**  
#### **Function:**  
```python
cv2.line(image, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8)
```
#### **Arguments:**  
- `image (numpy.ndarray)`: Input image.  
- `pt1 (tuple)`: Starting coordinates `(x1, y1)`.  
- `pt2 (tuple)`: Ending coordinates `(x2, y2)`.  
- `color (tuple)`: Line color `(B, G, R)`.  
- `thickness (int, optional)`: Line thickness (default is `1`).  
- `lineType (int, optional)`: Type of line:  
  - `cv2.LINE_4`: Basic line.  
  - `cv2.LINE_8`: Smooth line (default).  
  - `cv2.LINE_AA`: Anti-aliased line.  

#### **Example:**  
```python
import cv2

img = cv2.imread('sai.png')
img = cv2.resize(img, (300, 500))

# Draw a green line
cv2.line(img, (50, 50), (250, 50), (0, 255, 0), thickness=2, lineType=cv2.LINE_8)

cv2.imshow("Line Example", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **2. `cv2.rectangle()` - Drawing a Rectangle**  
#### **Function:**  
```python
cv2.rectangle(image, pt1, pt2, color, thickness=1)
```
#### **Arguments:**  
- `image (numpy.ndarray)`: Input image.  
- `pt1 (tuple)`: Top-left corner `(x1, y1)`.  
- `pt2 (tuple)`: Bottom-right corner `(x2, y2)`.  
- `color (tuple)`: Rectangle color `(B, G, R)`.  
- `thickness (int, optional)`: Border thickness (-1 for filled rectangle).  

#### **Example:**  
```python
cv2.rectangle(img, (50, 100), (250, 300), (0, 0, 255), thickness=3)
```

---

### **3. `cv2.circle()` - Drawing a Circle**  
#### **Function:**  
```python
cv2.circle(image, center, radius, color, thickness=1)
```
#### **Arguments:**  
- `image (numpy.ndarray)`: Input image.  
- `center (tuple)`: Center of the circle `(x, y)`.  
- `radius (int)`: Radius of the circle.  
- `color (tuple)`: Circle color `(B, G, R)`.  
- `thickness (int, optional)`: Border thickness (-1 for filled circle).  

#### **Example:**  
```python
cv2.circle(img, (150, 400), 40, (255, 0, 0), thickness=-1)
```

---

### **4. `cv2.ellipse()` - Drawing an Ellipse**  
#### **Function:**  
```python
cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness=1)
```
#### **Arguments:**  
- `image (numpy.ndarray)`: Input image.  
- `center (tuple)`: Center of the ellipse `(x, y)`.  
- `axes (tuple)`: Half-width and half-height `(major, minor)`.  
- `angle (int)`: Rotation angle of the ellipse.  
- `startAngle (int)`: Starting angle of the arc.  
- `endAngle (int)`: Ending angle of the arc.  
- `color (tuple)`: Ellipse color `(B, G, R)`.  
- `thickness (int, optional)`: Border thickness (-1 for filled ellipse).  

#### **Example:**  
```python
cv2.ellipse(img, (150, 200), (80, 50), 0, 0, 180, (255, 255, 0), thickness=2)
```

---

### **5. Complete Example: Drawing Lines, Rectangles, Circles, and Ellipses**
```python
import cv2

# Load and resize the image
img = cv2.imread('sai.png')
if img is not None:
    img = cv2.resize(img, (300, 500))

    # Draw a green line
    cv2.line(img, (50, 50), (250, 50), (0, 255, 0), thickness=2, lineType=cv2.LINE_8)

    # Draw a red rectangle
    cv2.rectangle(img, (50, 100), (250, 300), (0, 0, 255), thickness=3)

    # Draw a blue circle
    cv2.circle(img, (150, 400), 40, (255, 0, 0), thickness=-1)

    # Draw a yellow ellipse
    cv2.ellipse(img, (150, 200), (80, 50), 0, 0, 180, (255, 255, 0), thickness=2)

    # Display the image with shapes
    cv2.imshow("Bounding Box Example", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not load sai.png")
```

---

### **Summary**
| Function          | Purpose                 | Key Arguments |
|------------------|------------------------|--------------|
| `cv2.line()`     | Draws a line            | `pt1`, `pt2`, `color`, `thickness`, `lineType` |
| `cv2.rectangle()` | Draws a rectangle      | `pt1`, `pt2`, `color`, `thickness` |
| `cv2.circle()`   | Draws a circle         | `center`, `radius`, `color`, `thickness` |
| `cv2.ellipse()`  | Draws an ellipse       | `center`, `axes`, `angle`, `startAngle`, `endAngle`, `color`, `thickness` |

---

