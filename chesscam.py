import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os


CORNERS_FILE = "board_corners.json"


def save_corners(corners):
    """
    Save corner coordinates to a JSON file.
    
    Args:
        corners: List of 4 corner coordinates
    """
    with open(CORNERS_FILE, 'w') as f:
        json.dump(corners, f)
    print(f"Corners saved to {CORNERS_FILE}")


def load_corners():
    """
    Load corner coordinates from JSON file if it exists.
    
    Returns:
        List of 4 corner coordinates, or None if file doesn't exist
    """
    if os.path.exists(CORNERS_FILE):
        try:
            with open(CORNERS_FILE, 'r') as f:
                corners = json.load(f)
            return corners
        except:
            return None
    return None


def select_board_corners(image_path, prompt_for_new=False):
    """
    Interactively select 4 corners of the chess board.
    Uses saved corners silently, or prompts for new ones if requested.
    
    Args:
        image_path: Path to the image
        prompt_for_new: If True, ask user if they want to select new corners
        
    Returns:
        List of 4 corner coordinates as (x, y) tuples
    """
    # Check if saved corners exist
    saved_corners = load_corners()
    if saved_corners:
        if prompt_for_new:
            response = input(f"Saved corners found: {saved_corners}. Use them? (y/n): ")
            if response.lower() == 'y':
                return saved_corners
        else:
            return saved_corners
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title("Click 4 corners of the chess board (top-left, top-right, bottom-right, bottom-left)")
    ax.axis("off")
    
    corners = []
    
    def on_click(event):
        if event.inaxes != ax:
            return
        if len(corners) < 4:
            x, y = int(event.xdata), int(event.ydata)
            corners.append((x, y))
            ax.plot(x, y, 'ro', markersize=8)
            ax.text(x, y, f'  {len(corners)}', color='red', fontsize=10)
            fig.canvas.draw()
            print(f"Corner {len(corners)}: ({x}, {y})")
        
        if len(corners) == 4:
            print("All 4 corners selected!")
            plt.close()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    
    if len(corners) == 4:
        save_corners(corners)
        return corners
    
    return None


def warp_board_old(img, corners, board_size=800):
    """
    Warp board to standard view using perspective transform.
    
    Args:
        img: Input image
        corners: List of 4 corner coordinates
        board_size: Output board size in pixels
        
    Returns:
        Warped image
    """
    src_pts = np.array(corners, dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [board_size - 1, 0],
        [board_size - 1, board_size - 1],
        [0, board_size - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (board_size, board_size))
    
    return warped

def warp_board(img, corners, board_size=800, margin=100):

    src_pts = np.array(corners, dtype=np.float32)

    dst_pts = np.array([
        [margin, margin],
        [board_size - margin - 1, margin],
        [board_size - margin - 1, board_size - margin - 1],
        [margin, board_size - margin - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        src_pts,
        dst_pts
    )

    warped = cv2.warpPerspective(
        img,
        matrix,
        (board_size, board_size)
    )

    return warped


def detect_changed_squares(warped1, warped2, board_size=800, squares=8, intensity_weighting=0.5):
    """
    Detect movement regions by finding the 2 largest unconnected components in grayscale difference.
    Centroids can be weighted by intensity of differences.
    
    Args:
        warped1, warped2: Warped board images
        board_size: Size of warped board
        squares: Number of squares per side (default 8 for chess)
        intensity_weighting: Balance between standard (0) and intensity-weighted (1) centroid. Default 0.5
        
    Returns:
        List of (x, y) centroids (max 2), grayscale difference
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray1 = clahe.apply(gray1)
    gray2 = clahe.apply(gray2)
    
    # Grayscale difference
    total_gray_diff = cv2.absdiff(gray1, gray2)
    total_gray_diff = cv2.GaussianBlur(total_gray_diff, (5, 5), 0)  # Add this line
    
    # Threshold
    _, binary_diff = cv2.threshold(total_gray_diff, 50, 255, cv2.THRESH_BINARY)  # Up from 50
    
    # Morphological filtering to enhance connected regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
    binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)
    
    # Connected component analysis - find distinct movement regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_diff, connectivity=8
    )
    
    # Get all components sorted by size (largest first)
    components = []
    for label_idx in range(1, num_labels):  # Skip background (0)
        area = stats[label_idx, cv2.CC_STAT_AREA]
        components.append((area, label_idx))
    
    # Sort by area descending
    components.sort(reverse=True)
    
    # Get top 2 largest components (representing max 1 move = 2 regions)
    top_2_components = components[:2]
    
    # Extract centroids with optional intensity weighting
    movement_centroids = []
    for area, label_idx in top_2_components:
        # Get standard centroid
        cx_standard, cy_standard = centroids[label_idx]
        
        # Calculate intensity-weighted centroid
        if intensity_weighting > 0:
            # Get mask for this component
            component_mask = (labels == label_idx).astype(np.uint8)
            # Weighted by intensity: multiply mask by intensity values
            weighted_component = total_gray_diff * component_mask
            # Use moments to calculate intensity-weighted centroid
            moments = cv2.moments(weighted_component)
            if moments['m00'] > 0:
                cx_weighted = moments['m10'] / moments['m00']
                cy_weighted = moments['m01'] / moments['m00']
            else:
                cx_weighted = cx_standard
                cy_weighted = cy_standard
            
            # Blend: 0 = standard, 1 = fully weighted by intensity
            cx = cx_standard * (1 - intensity_weighting) + cx_weighted * intensity_weighting
            cy = cy_standard * (1 - intensity_weighting) + cy_weighted * intensity_weighting
        else:
            cx, cy = cx_standard, cy_standard
        
        movement_centroids.append((int(cx), int(cy)))
    
    return movement_centroids, total_gray_diff


def compare_images_with_board(image_path1, image_path2, intensity_weighting=0.5, margin=100):
    """
    Compare 2 images by warping to board space and detecting changed squares.
    Uses grayscale difference with connected component analysis.
    
    Args:
        image_path1, image_path2: Paths to the 2 images
        intensity_weighting: Parameter to weight centroids (0=geometric center, 1=intensity peak). Default 0.5
    """
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        print("Error: Could not load both images")
        return
    
    # Select corners from first image (silent unless needed)
    corners = select_board_corners(image_path1, prompt_for_new=False)
    if corners is None:
        print("Corner selection failed")
        return
    
    board_size = 800
    
    # Warp both images
    warped1 = warp_board(img1, corners, board_size)
    warped2 = warp_board(img2, corners, board_size)
    
    # Detect movement regions with intensity weighting
    movement_centroids, gray_diff = detect_changed_squares(warped1, warped2, board_size, intensity_weighting=intensity_weighting)
    
    # Visualize movement centroids on both warped images
    warped1_rgb = cv2.cvtColor(warped1, cv2.COLOR_BGR2RGB)
    warped2_rgb = cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB)
    
    # Create subplot with both images
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display image 1 with centroids
    axes[0].imshow(warped1_rgb)
    for i, (cx, cy) in enumerate(movement_centroids, 1):
        axes[0].plot(cx, cy, 'r*', markersize=20, label=f"Region {i}")
    axes[0].set_title("Image 1 - Movement Regions")
    axes[0].legend(loc='upper right')
    axes[0].axis("off")
    
    # Display image 2 with centroids
    axes[1].imshow(warped2_rgb)
    for i, (cx, cy) in enumerate(movement_centroids, 1):
        axes[1].plot(cx, cy, 'r*', markersize=20, label=f"Region {i}")
    axes[1].set_title("Image 2 - Movement Regions")
    axes[1].legend(loc='upper right')
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Print centroids
    print(f"\nDetected {len(movement_centroids)} movement regions:")
    for i, (cx, cy) in enumerate(movement_centroids, 1):
        # Convert pixel coordinates to board position
        # col = int(cx // (board_size // 8))
        # row = int(cy // (board_size // 8))

        effective_board_size = (
            board_size
            - 2 * margin
        )

        col = int(
            (cx - margin)
            / (effective_board_size / 8)
        )

        row = int(
            (cy - margin)
            / (effective_board_size / 8)
        )




        col = min(7, max(0, col))
        row = min(7, max(0, row))
        col_letter = chr(ord('a') + col)
        row_number = 8 - row
        print(f"  Region {i}: pixel ({cx}, {cy}) -> board square {col_letter}{row_number}")


if __name__ == "__main__":
    # Example usage
    for ind in range(6):
        compare_images_with_board(
            f"chessims_opening/im{ind}.jpeg",
            f"chessims_opening/im{ind + 1}.jpeg"
        )





