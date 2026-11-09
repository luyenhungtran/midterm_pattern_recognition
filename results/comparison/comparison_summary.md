# Nhận xét so sánh MNIST

**Kiểm tra đạt:** đủ 4 phương pháp × 3 tập; độ đo khớp confusion matrix và support theo lớp. Chỉ số train/validation trùng cả thứ tự giữa Gaussian NB, MLP, Stacking AdaBoost. Softmax chưa lưu split_indices; chưa thể xác nhận danh tính ảnh của split này chỉ bằng metadata. Các phép kiểm tra này không chứng minh cùng phần cứng hoặc cùng ngân sách tuning.

**Phạm vi:** so sánh các pipeline đã xây dựng, không phải benchmark có ngân sách tìm kiếm ngang nhau. MLP tự viết ở 04 khác MLP256_128 của bộ ba 05/06; không gộp hai mô hình này. Validation đã dùng chọn cấu hình/checkpoint. OOF chỉ ngăn meta-model thấy dự đoán in-sample khi huấn luyện; không biến validation đã tuning thành tập đánh giá độc lập.

**Nhận xét:** Stacking AdaBoost đạt test accuracy cao nhất trong bốn pipeline (98.54%), với 146 ảnh sai. MLP tự viết có train accuracy 100.00% và validation 98.30%, chênh 1.70 điểm phần trăm. Đây là dấu hiệu chênh lệch khớp train và tổng quát hóa, chưa đủ xác định nguyên nhân. Train của stacking được dự đoán bằng base models refit, không phải điểm OOF độc lập.

**Nhận xét:** so với MLP tự viết, stacking tăng 0.35 điểm phần trăm accuracy, nhưng thời gian fit ghi nhận gấp 3.3 lần và dự đoán gấp 1456.9 lần. Các tỷ số mô tả các phép đo đã lưu, chưa tách ảnh hưởng môi trường. Không có mô hình tốt nhất đồng thời cho mọi nhu cầu.

**Nhận xét:** Gaussian NB: lớp 5 có recall thấp nhất 51.35%; Softmax: lớp 8 có recall thấp nhất 86.76%; MLP: lớp 3 có recall thấp nhất 97.52%; Stacking AdaBoost: lớp 7 có recall thấp nhất 97.86%. Recall thấp là tỷ lệ bỏ sót cao trong lớp thật; cặp nhầm có số đếm lớn phải được đọc cùng support. Confusion matrix mô tả lỗi, chưa chứng minh nét chữ hay đặc trưng nào gây ra lỗi.

**Nhận xét:** MLP sửa đúng 607 ảnh Softmax dự đoán sai, nhưng làm sai thêm 41 ảnh Softmax từng đúng; giảm ròng 566 lỗi. Accuracy cao hơn không có nghĩa mô hình mới đúng trên mọi ảnh mà mô hình cũ từng đúng.

**Nhận xét:** các cấu hình cuối không đồng nhất hoàn toàn, vì vậy kết quả này là đối chiếu thực nghiệm, không phải kiểm tra tương đương từng bước. Gaussian có thí nghiệm matched epsilon riêng; không viết rằng hai cấu hình cuối dùng cùng smoothing. MLP cuối dùng L2 khác nhau, không phải L2=0.

**Nhận xét:** thay đổi siêu tham số có thể tạo khác biệt lớn, đặc biệt smoothing của Gaussian và learning rate của meta-model. Mỗi đường giữ các điều kiện theo notebook nguồn; không so trực tiếp giá trị tham số giữa các họ mô hình. Giá trị tốt nhất của một sweep không nhất thiết là cấu hình cuối đang báo cáo, ví dụ Gaussian giữ cấu hình cuối đã chốt trước đó.

**Stacking so với SVM:** accuracy chênh +0.21 điểm phần trăm; sửa đúng 40, làm sai thêm 19, giảm ròng 21 lỗi. Fit ghi nhận 6.91 lần, dự đoán 1.08 lần so với mốc. Chênh lệch test chỉ mô tả lượt chạy này, chưa chứng minh khác biệt có ý nghĩa thống kê.

**Stacking so với Soft voting:** accuracy chênh +0.02 điểm phần trăm; sửa đúng 20, làm sai thêm 18, giảm ròng 2 lỗi. Fit ghi nhận 4.60 lần, dự đoán 1.00 lần so với mốc. Chênh lệch test chỉ mô tả lượt chạy này, chưa chứng minh khác biệt có ý nghĩa thống kê.

**Mốc tham chiếu:** SVM được chọn trên validation của bộ ba. Bảng lỗi lấy từ 06 và đã kiểm tra tính nhất quán với số lỗi/accuracy; 06 chưa lưu dự đoán stacking theo từng ảnh nên chưa tái dựng độc lập được danh sách ảnh sửa đúng/làm sai. Soft voting và weighted voting không phải hai nguồn bằng chứng độc lập nếu cùng trọng số.

**Thí nghiệm 08:** với 50 epoch, CNN_FC128 có 206,922 tham số, MLP260 có 206,710; CNN tăng 1.13 điểm phần trăm test accuracy. Số tham số gần nhau hỗ trợ so sánh kiến trúc nhưng không đồng nghĩa số phép tính hoặc thời gian bằng nhau.

**Thí nghiệm 09:** CNN_FC128: thay đổi -0.33 điểm phần trăm; MLP260: thay đổi -8.35 điểm phần trăm. Đây là clean test của cấu hình xoay/dịch/nhiễu đang dùng; không suy rộng thành augmentation luôn tốt hoặc luôn xấu. Đọc thêm phép đo ảnh biến đổi bên dưới trước khi kết luận robustness. Không ghép trực tiếp với bảng 50 epoch của 08.

**Robustness tại mức biến đổi lớn nhất đã đo:** CNN_FC128, rotation=30: augmentation chênh +15.04 điểm phần trăm; CNN_FC128, shift=4: augmentation chênh +43.62 điểm phần trăm; MLP260, rotation=30: augmentation chênh +11.32 điểm phần trăm; MLP260, shift=4: augmentation chênh +57.36 điểm phần trăm. Đọc cùng clean accuracy ở mức 0: giảm điểm clean và tăng điểm trên ảnh biến đổi có thể cùng xảy ra. Các số liệu chỉ áp dụng cho phép biến đổi và mức độ trong thí nghiệm 09.

**Kết luận cho báo cáo:** bốn pipeline có sự đánh đổi giữa chất lượng, chi phí và mức độ giải thích. Gaussian NB đạt 77.53%, Softmax 92.53%, MLP tự viết 98.19%, stacking 98.54% test accuracy. Bảng lỗi ghép cặp cho biết lợi ích và thiệt hại trên dự đoán, thay vì chỉ nhìn accuracy tổng. Khi lựa chọn cho ứng dụng, dùng validation và ràng buộc thời gian đã xác định trước, không chọn lại bằng test. Hạn chế: một seed, tuning khác ngân sách, thời gian chưa đo lặp trong môi trường thống nhất và thiếu ID ảnh ở một số đầu ra. Các nhận định nguyên nhân như độc lập pixel, khác phân phối OOF/refit hay augmentation quá mạnh là giả thuyết cần thí nghiệm kiểm chứng.