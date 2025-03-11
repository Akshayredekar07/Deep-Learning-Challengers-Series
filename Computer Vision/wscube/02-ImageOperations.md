| **Operation**      | **Description**                              | **Use Case**                          |  
|--------------------|--------------------------------|----------------------------------|  
| **Bitwise AND**    | Keeps common white areas in two images  | Masking, ROI extraction       |  
| **Bitwise OR**     | Merges white areas from both images    | Combining masks, overlays     |  
| **Bitwise XOR**    | Keeps only non-overlapping white areas | Difference detection          |  
| **Bitwise NOT**    | Inverts colors (black ↔ white)        | Negative images, mask inversion |  
| **`np.hstack()`**  | Horizontally stacks images             | Side-by-side comparison       |  
| **`np.vstack()`**  | Vertically stacks images               | Multi-row visualization       |  