import numpy as np
import cv2
import alphashape

# 啟用鏡頭
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 創建視窗
cv2.namedWindow('oxxostudio')

n = 1                      # 照片起始編號
is_pressed = False
font = cv2.FONT_HERSHEY_SIMPLEX  # 字體
area_text = "Area: -"  # 初始面積文字

while True:
    ret, img = cap.read()               # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        break
    height, width = img.shape[:2]

    # 計算裁切的像素數
    crop_percentage = 0.1
    crop_width = int(width * crop_percentage)
    crop_height = int(height * crop_percentage)

    # 計算裁切後的區域
    left = crop_width
    top = crop_height
    right = width - crop_width
    bottom = height - crop_height

    # 裁切影像
    img = img[top:bottom, left:right]
    # 轉換顏色為 BGRA
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    key = cv2.waitKey(1)
    
    if key == 32:  # 按下空白就截圖存檔
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract black pixels' positions
        points = np.column_stack(np.where(gray <= 90))

        # Convert points to (x, y) format
        points = points[:, ::-1]  # Swap columns to (x, y)

        # Sample points (if needed)
        sample_size = int(0.1 * len(points))
        if sample_size > 0:
            points = points[np.random.choice(points.shape[0], sample_size, replace=False)]
        points = points.astype(np.float64)

        # Calculate center point and normalize points
        center_point = points.mean(axis=0)
        max_val = int(np.max(np.abs(points))) + 1
        points = points / max_val

        # Compute alpha shape
        alpha_shape = alphashape.alphashape(points, lambda ind, r: 1.0 + any(np.array(points)[ind][:, 0] == 0.0))
        #alpha_shape = alphashape.alphashape(points,1)
        if alpha_shape:
            area = alpha_shape.area * max_val * max_val*118.4/47937.4
            area_text = f"Area: {area:.2f} cm2"

        # Draw alpha shape on original image
        if alpha_shape:
            # Create a copy of the original image to draw on
            overlay = img.copy()

            # Convert alpha shape coordinates to integer for drawing
            alpha_shape_coords = (np.array(alpha_shape.exterior.coords)) * max_val
            alpha_shape_coords = alpha_shape_coords.astype(int)

            # Draw alpha shape as a filled polygon
            cv2.fillPoly(overlay, [alpha_shape_coords], (0, 0, 255))  # Red color for example

            # Blend the overlay with the original image
            alpha = 0.3  # Transparency level
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            is_pressed = True

    if is_pressed:
        overlay = img.copy()
        cv2.fillPoly(overlay, [alpha_shape_coords], (0, 0, 255))  # Red color for example

        # Blend the overlay with the original image
        alpha = 0.3  # Transparency level
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 將面積信息顯示在視窗上
    cv2.putText(img, area_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('oxxostudio', img)  # 顯示圖片

    # 檢查視窗是否關閉或按下了 'q' 鍵
    if key == ord('q'):
        break

cap.release()  # 釋放資源
cv2.destroyAllWindows()
