import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
# tăng màu đen của ảnh
def adjust_brightness_gray(imgColor):
    if(90<imgColor.mean()<170):
        img_gray_lp = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
        _, imgColor = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imgColor= cv2.erode(imgColor, (3, 3)) # type: ignore
        imgColor = cv2.dilate(imgColor, (3, 3)) # type: ignore
                         
    else:
        imgColor=cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
        # Lấy chiều cao và chiều rộng của ảnh
        height, width = imgColor.shape
        # Duyệt qua từng pixel trong ảnh
        for i in range(height):
            for j in range(width):
                pixel_value = imgColor[i, j]

                # Nếu pixel có giá trị trên 200, tăng 10
                if pixel_value < 200:
                    imgColor[i, j] -= 60 
    return imgColor
# hiển thị hình ảnh dưới dạng ảnh màu
def showImgRGBMode(img,TitleIMG): 
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # type: ignore
    plt.title(TitleIMG)
    plt.axis('off')
    plt.show()
# hiển thị hình ảnh dưới dạng ảnh xám
def showImgGrayMode(imgGray,TiltleIMG):
    plt.figure(figsize=(10, 5))
    plt.imshow(imgGray, cmap='gray')
    plt.title(TiltleIMG)
    plt.axis('off')
    plt.show()
#  Làm mịn và giảm nhiễu bằng GaussianBlur
def ReduceNoiceAndSmooth(imgGray):
    blurred_image = cv2.GaussianBlur(imgGray, (7, 7), 1.4)
    return blurred_image
# Tăng độ tương phản và lọc trung vị để loại bỏ nhiễu
def IncreaseContrastAndMedianFilter(imgBlurred):
    # cân bằng Histogram - tăng cường độ tương phản bằng equalize
    equalized = cv2.equalizeHist(imgBlurred)
    # lọc trung vị để loại bỏ nhiễu sử dụng MedianBlur
    denoised = cv2.medianBlur(equalized, 7)
    return denoised
# nhị phân hóa hình ảnh 
def BinarizeImg(imgGray):
    ADAPTIVE_THRESH_BLOCK_SIZE = 19
    ADAPTIVE_THRESH_WEIGHT = 9
    imgThresh = cv2.adaptiveThreshold(imgGray, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return imgThresh
# phát hiện biên bằng Dilation
def GradientMO(BinaryImg):
    # Tạo kernel cho các phép toán hình thái học
    kernel = np.ones((3,3), np.uint8)

    # Giãn nở và xói mòn
    dilation = cv2.dilate(BinaryImg, kernel, iterations=1)
    erosion = cv2.erode(BinaryImg, kernel, iterations=1)

    # Tính gradient hình thái học
    morph_gradient = cv2.morphologyEx(BinaryImg, cv2.MORPH_GRADIENT, kernel)
    return morph_gradient
# phát hiện biên bằng Sobel operator bằng cách tìm vị trí có độ tương phản diễn ra mạnh nhất
def SobelOperator(DilatedImg):
    # sử dụng Sobel operator để tìm biên mà tại đó điểm tương phản là mạnh nhất

    # Sử dụng Sobel Operator để tìm biên
    sobelx = cv2.Sobel(DilatedImg, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(DilatedImg, cv2.CV_64F, 0, 1, ksize=5)

    # Tính độ lớn của gradient
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Chuẩn hóa về phạm vi từ 0 đến 255 và chuyển đổi sang kiểu uint8
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(np.uint8)
    return gradient_magnitude
def EdgeDetectionWithCanny(GradientImg):
    canny_image = cv2.Canny(GradientImg, 250, 255)
    return canny_image

def chuyen_den_trang(image):
    # Chuyển ảnh màu sang ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Áp dụng ngưỡng nhị phân để tạo ra ảnh đen trắng
    _, black_white_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # cộng nền trắng 
    #  _, black_white_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # 
    
    return black_white_image

def dao_den_thanh_trang(image):
    # Đảo ảnh nhị phân bằng phép toán bitwise NOT
    inverted_image = cv2.bitwise_not(image)
    return inverted_image
def add_padding(image):
    height, width = image.shape[:2]
    target_size = max(height, width)
    squared_image = np.ones((target_size, target_size), dtype=np.uint8) * 255
    x_offset = (target_size - width) // 2
    y_offset = (target_size - height) // 2
    squared_image[y_offset:y_offset+height, x_offset:x_offset+width] = image
    return squared_image
def merge_arrays(a, b):
    result = a.copy()  # Tạo một bản sao của mảng a để không ảnh hưởng đến mảng gốc
    result.extend(b)   # Sử dụng phương thức extend() để ghép mảng b vào mảng a
    return result


# Thiết lập các tham số như kích cỡ cảu bộ lọc Gauss 
# Bộ lọc gauss ( làm mở và giảm nhiều 1  dạng của convolution (tích chập) giảm những tác nhân gây nhiễu gây mờ bằng cách dauwj vào các điểm ảnh đậm màu)
GAUSIAN_SMOOTH_FILTER_SIZE = (5,5) # ở đây ma trận chập sẽ là 5*5
# kích thước của ngưỡng thích nghi (Đây là kích thước của khu vực lân cận sẽ được sử dụng để tính ngưỡng cục bộ. Trong trường hợp này, kích thước của khu vực là 19x19 pixel.)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
# trọng số ngưỡng thích nghi (Đây là một hệ số được sử dụng trong phương pháp tính toán ngưỡng cục bộ. Nó ảnh hưởng đến mức độ làm mờ và làm sắc nét ảnh kết quả)
ADAPTIVE_THRESH_WEIGHT = 9


# hàm tiền xử lý ảnh gốc (
# mục tiêu là : trích xuất giá trị cường độ sáng, tối đa hóa tương phản và làm mịn ảnh)
def tienxuly(img):
    # trích xuất bàng hàm trích xuất cường độ sáng
    anh_xam = trich_xuat_gia_tri_cuong_do_sang(img)
    toidahoa = toi_da_hoa_anh(anh_xam) # ảnh màu xám
    # lưu vào file trong folder result  tên là anhsaucuonghoa.jpg
    cv2.imwrite("result/anhsaucuonghoa.png", toidahoa)
    dai, rong = anh_xam.shape
    anh_sau_khi_lam_min = cv2.GaussianBlur(toidahoa,GAUSIAN_SMOOTH_FILTER_SIZE,1.4)
    # tạo ảnh nhị phân
    anh_nhi_phan = cv2.adaptiveThreshold(anh_sau_khi_lam_min,255.0,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,ADAPTIVE_THRESH_BLOCK_SIZE,ADAPTIVE_THRESH_WEIGHT)
    return anh_xam, anh_nhi_phan

# hàm trích xuất cường độ sáng 
# mục đích là : 
# Hàm trích xuất giá trị cường độ sáng từ ảnh gốc bằng cách chuyển đổi từ không gian màu BGR sang HSV và tách lấy kênh giá trị cường độ sáng    
def trich_xuat_gia_tri_cuong_do_sang(img):
    cao, rong, numChannels = img.shape
    # chuyển đôi ảnh từ không gian màu (BGR) ==> HSV(vì giá trị này thường phản ánh độ sáng của mỗi pixel mà không bị ảnh hưởng bởi màu sắc)
    # Hệ màu HSV dùng để xử lý màu rất thuận tiện, vì mỗi màu sẽ có 1 giá trị Hue [0;360]
    # Hình ảnh đa số cấu tạo từ 3 kênh màu Red-Green-Blue
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # tách 3 tông màu của HSV ( hue , saturation, value)
    # Trong không gian màu HSV, giá trị (Value) đại diện cho cường độ sáng của một pixel, trong khi Hue đại diện cho màu sắc và Saturation đại diện cho độ bão hòa màu
    hue, saturation, value = cv2.split(imgHSV)
    return value #(cường độ sáng của một pixel)


# Hàm tối đa hóa độ tương phản của ảnh xám bằng cách áp dụng các phép toán hình thái học Top-Hat và Black-Hat.
# mục đích là làm vùng tối thì tối hơn vung trắng thì càn trắng hơn
def toi_da_hoa_anh(img):
    cao, rong = img.shape
    # Tạo bộ lọc hình chữ nhật
    # được sử dụng để tạo ra các phần tử cấu trúc (structuring elements) cho các phép biến đổi hình thái học,
    # chẳng hạn như 
    # giãn nở (dilation) cho đối tượng ban đầu trong ảnh tăng lên về kích thước 
    # co lại (erosion) giảm kích thước của đối tượng, tách rời các đối tượng gần nhau, làm mảnh và tìm xương của đối tượng
    # mở (opening) xóa các điểm ảnh nhiều xung quanh hình
    # đóng (closing). xóa các điểm nhiễu bên trong hình

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # tophat (phép trừ ảnh của ảnh ban đầu với ảnh sau khi thực hiện phép mở) ==> nổi bật nhưng chi tiết trắng trong nền tối
    img_top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, structuringElement, iterations=10) # số lần lặp lại cho phép biến đổi hình thái học iterations=10
    cv2.imwrite("result/anhsautophat.png", img_top_hat)
    
    # backhat (Nổi bật chi tiết tối trong nền sáng)
    img_back_hat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, structuringElement, iterations=10)
    cv2.imwrite("result/anhsaubackhat.png", img_back_hat)
    # áp dụng công thức + top - back:
    img_sau_chinh = cv2.add(img,img_top_hat)
    img_sau_chinh = cv2.subtract(img_sau_chinh,img_back_hat)

    # hiện ảnh sau chỉnh
    
    return img_sau_chinh


      