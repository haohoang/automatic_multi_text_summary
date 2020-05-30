# automatic_multi_text_summary

- Dữ liệu DUC2007
- Thuật toán gồm Kmeans, LSA, MMR theo https://drive.google.com/file/d/1-h6Wtab3_RumMXAihVMJl5ur9fIM63dc/view?usp=sharing.
- Các bước thực hiện
  + Ghép các văn bản trong cùng một chủ đề làm một
  + Tách câu
  + Biểu diễn các câu dưới dạng vector
  + Phân cụm các câu bằng Kmeans
  + Chọn câu trung tâm của từng cụm làm ứng cử viên cho bản tóm tắt
  + Sau khi phân cụm sẽ có những cụm chứa ít thông tin nên dùng LSA để loại bỏ những cụm đó
  + Sử dụng MMR để loại bỏ những câu dư thừa ra khỏi bản tóm tắt
  + Sắp xếp các câu được chọn theo vị trí của câu đó trong văn bản gốc.
