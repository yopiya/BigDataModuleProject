from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import count
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors, utils
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt
import pandas as pd
import math
from PyPDF2 import PdfMerger

print("เริ่มการทำงาน...")
# สร้าง SparkSession
spark = SparkSession.builder.appName("CustomerSegmentation").getOrCreate()
print("กำลังอ่านข้อมูล")
# อ่านข้อมูลจากไฟล์ customer_data.csv และสร้าง DataFrame
data = spark.read.csv("customer_data.csv", header=True, inferSchema=True)

# เตรียมข้อมูลในรูปแบบที่เหมาะสมสำหรับการใช้ KMeans algorithm โดยใช้ VectorAssembler
feature_columns = ["Salary", "total_asset", "Amount_Paid", "Pay_now_total", "Num_Missed_Pay", "Total_Installments"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data_with_features = assembler.transform(data)

# ทำการ Standardize ข้อมูลก่อนการแบ่งกลุ่ม
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(data_with_features)
data_with_scaled_features = scaler_model.transform(data_with_features)
print("สร้างโมเดล")
# สร้างโมเดล KMeans
k = 4  # จำนวนกลุ่มที่ต้องการสร้าง
kmeans = KMeans(k=k, seed=42)
model = kmeans.fit(data_with_scaled_features)
print("เริ่มการทำนาย")
# ทำนายกลุ่มสำหรับแต่ละลูกค้า
predictions = model.transform(data_with_scaled_features)

# นับจำนวนลูกค้าในแต่ละกลุ่ม
customer_counts = predictions.groupBy("prediction").agg(count("*").alias("customer_count"))
print("แสดงกลุ่มลูกค้า")
# แสดงข้อมูลจำนวนลูกค้าในแต่ละกลุ่ม
customer_counts.show()
print("แสดงค่า k-mean")
# แสดงค่า k-mean ของแต่ละกลุ่ม
centroids = model.clusterCenters()
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: {centroid}")

print("บันทึกส่งออกไฟล์กลุ่ม 1")
# กลุ่มที่ 1
customers_in_group_1 = predictions.filter(predictions.prediction == 0)
customers_in_group_1_pd = customers_in_group_1.toPandas()
customers_in_group_1_pd.to_csv("customers_group_1.csv", index=False)

print("บันทึกส่งออกไฟล์กลุ่ม 2")
# กลุ่มที่ 2
customers_in_group_2 = predictions.filter(predictions.prediction == 1)
customers_in_group_2_pd = customers_in_group_2.toPandas()
customers_in_group_2_pd.to_csv("customers_group_2.csv", index=False)

print("บันทึกส่งออกไฟล์กลุ่ม 3")
# กลุ่มที่ 3
customers_in_group_3 = predictions.filter(predictions.prediction == 2)
customers_in_group_3_pd = customers_in_group_3.toPandas()
customers_in_group_3_pd.to_csv("customers_group_3.csv", index=False)

print("บันทึกส่งออกไฟล์กลุ่ม 4")
# กลุ่มที่ 4
customers_in_group_4 = predictions.filter(predictions.prediction == 3)
customers_in_group_4_pd = customers_in_group_4.toPandas()
customers_in_group_4_pd.to_csv("customers_group_4.csv", index=False)

print("เริ่มการพล็อตกราฟ ")
# พล็อตกราฟจำนวนลูกค้าในแต่ละกลุ่ม
customer_counts_pd = customer_counts.toPandas()
plt.bar(customer_counts_pd["prediction"], customer_counts_pd["customer_count"])
plt.xlabel("Group")
plt.ylabel("Number of Customers")
plt.title("Number of Customers in Each Group")
plt.savefig("customer_counts_graph.png")

# พล็อตกราฟแสดงค่า k-mean ของแต่ละกลุ่ม
for i, centroid in enumerate(centroids):
    plt.bar(i, centroid, label=f"Group {i}")
plt.xlabel("Group")
plt.ylabel("K-Mean Value")
plt.title("K-Mean Value of Each Group")
plt.legend()
plt.savefig("kmean_values_graph.png")

print("เริ่มการสร้างตาราง")
customer_counts_pd = customer_counts.toPandas()

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=customer_counts_pd.values,
                 colLabels=customer_counts_pd.columns,
                 cellLoc='center',
                 loc='top',
                 colWidths=[0.5] * len(customer_counts_pd.columns))
table.auto_set_font_size(False)
table.set_fontsize(10)  # กำหนดขนาดตัวอักษรเท่ากับ 10
table.auto_set_column_width(col=list(range(len(customer_counts_pd.columns))))
plt.savefig('customer_counts_table.png', bbox_inches='tight')


# ทำตาราง centroid ของแต่ละกลุ่ม

centroids = model.clusterCenters()
centroids_data = [[f"Centroid {i+1}"] + centroid.tolist() for i, centroid in enumerate(centroids)]

# สร้างตาราง Pandas DataFrame สำหรับ Centroid
centroid_columns = ["Centroid"] + feature_columns
centroid_df = pd.DataFrame(centroids_data, columns=centroid_columns)

# แสดงตาราง Centroid
#print(centroid_df)

print("เริ่มการสร้างรูปภาพ")

# สร้างรูปภาพแสดงตาราง Centroid
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=centroid_df.values,
                 colLabels=centroid_df.columns,
                 cellLoc='left',
                 loc='top',
                 colWidths=[0.5] * len(centroid_df.columns))
table.auto_set_font_size(False)
table.set_fontsize(10)  # กำหนดขนาดตัวอักษรเท่ากับ 10
table.auto_set_column_width(col=list(range(len(centroid_df.columns))))  # กำหนดความกว้างของคอลัมน์ให้พอดีกับข้อความ
plt.savefig('centroid_table.png', bbox_inches='tight')

print("เริ่มการสร้างไฟล์แยกลูกค้า")

# อ่านข้อมูลจากไฟล์ของแต่ละกลุ่ม
group_1_data = pd.read_csv("customers_group_1.csv")
group_2_data = pd.read_csv("customers_group_2.csv")
group_3_data = pd.read_csv("customers_group_3.csv")
group_4_data = pd.read_csv("customers_group_4.csv")

# แปลงข้อมูลใน DataFrame เป็นรูปแบบของ List สำหรับใช้ในการสร้างตาราง
group_1_table_data = [group_1_data.columns.tolist()] + group_1_data.values.tolist()
group_2_table_data = [group_2_data.columns.tolist()] + group_2_data.values.tolist()
group_3_table_data = [group_3_data.columns.tolist()] + group_3_data.values.tolist()
group_4_table_data = [group_4_data.columns.tolist()] + group_4_data.values.tolist()

print("เริ่มการสร้างเอกสาร")

# นับจำนวนลูกค้าในแต่ละกลุ่ม
customer_counts = predictions.groupBy("prediction").agg(count("*").alias("customer_count"))

# พล็อต Scatter Plot จากค่า centroid ของแต่ละกลุ่ม
centroids = model.clusterCenters()
x = [centroid[0] for centroid in centroids]
y = [centroid[1] for centroid in centroids]

plt.scatter(x, y, color='red', label='Centroids')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Centroids")
plt.legend()
plt.savefig("centroids_scatter_plot.png")

# สร้างเอกสาร PDF
pdf_file = "customer_group_tables.pdf"

# ปรับขนาดหน้ากระดาษเป็นแนวนอน
doc = SimpleDocTemplate(pdf_file, pagesize=landscape(letter))

# สร้างรูปแบบตาราง
table_style = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),  # สีพื้นหลังสำหรับหัวตาราง
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),  # สีตัวอักษรสำหรับหัวตาราง
    ("ALIGN", (0, 0), (-1, -1), "LEFT"),  # จัดแนวตารางชิดซ้าย
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),  # แบบอักษรสำหรับหัวตาราง
    ("FONTSIZE", (0, 0), (-1, 0), 12),  # ขนาดตัวอักษรสำหรับหัวตาราง
    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),  # ระยะห่างด้านล่างสำหรับหัวตาราง
    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),  # สีพื้นหลังสำหรับข้อมูลในตาราง
    ("GRID", (0, 0), (-1, -1), 1, colors.black),  # เส้นตาราง
])

# สร้างเนื้อหาในเอกสาร
content = []

# สร้างตารางสำหรับแต่ละกลุ่ม
group_1_table = Table(group_1_table_data, style=table_style, colWidths=[70, 70, 70])
group_2_table = Table(group_2_table_data, style=table_style, colWidths=[100, 150, 120])
group_3_table = Table(group_3_table_data, style=table_style, colWidths=[100, 150, 120])
group_4_table = Table(group_4_table_data, style=table_style, colWidths=[100, 150, 120])


# กำหนดความกว้างของคอลัมน์ในตารางสำหรับแต่ละกลุ่ม
col_widths = [70, 70, 70]  # ปรับค่าตามต้องการ

group_1_table._argW = col_widths * len(group_1_table_data[0])
group_2_table._argW = col_widths * len(group_2_table_data[0])
group_3_table._argW = col_widths * len(group_3_table_data[0])
group_4_table._argW = col_widths * len(group_4_table_data[0])

# เพิ่มตารางลงในเอกสาร
content = [group_1_table, group_2_table, group_3_table, group_4_table]

# เพิ่มเนื้อหาลงในเอกสาร
doc.build(content)

print("เริ่มการสร้างเอกสาร PDF และเพิ่มรูปภาพลงในไฟล์")


# สร้างไฟล์ PDF และเพิ่มรูปภาพลงในไฟล์
pdf_file = 'customer_segmentation_report.pdf'
c = canvas.Canvas(pdf_file, pagesize=letter)
# Page 1
c.setFont("Helvetica-Bold", 16)
c.drawCentredString(300, 730, "Customer by Cluster")

c.drawImage("customer_counts_graph.png", 50, 450, 400, 300)

c.setFont("Helvetica", 12)
c.drawCentredString(250, 400, "Customer by Cluster")

c.setFont("Helvetica-Bold", 16)
c.drawCentredString(300, 330, "K-Mean Value by any Group")

c.drawImage("kmean_values_graph.png", 50, 100, 400, 300)

c.setFont("Helvetica", 12)
c.drawCentredString(250, 50, "K-Mean Value by any Group")

c.showPage()

# Page 2
c.drawImage("centroid_table.png", 20, 450, 550, 300)

c.setFont("Helvetica", 12)
c.drawCentredString(300, 650, "Centroid by any Group")

c.setFont("Helvetica-Bold", 16)
c.drawCentredString(300, 350, "Prediction Value")

c.drawImage("centroid_table.png", 100, 400, 300, 200)

c.save()

print("เริ่มการรวมเอกสาร PDF")

# เปิดไฟล์ PDF และสร้างอ็อบเจ็กต์ PdfFileMerger
pdf_merger = PdfMerger()

# เพิ่มไฟล์ PDF ที่ต้องการต่อกัน
pdf_merger.append('customer_segmentation_report.pdf')
pdf_merger.append('customer_group_tables.pdf')

# บันทึกเอกสาร PDF ที่ต่อกันไว้ในไฟล์ใหม่
pdf_merger.write('FullReport.pdf')

# ปิดอ็อบเจ็กต์ PdfFileMerger
pdf_merger.close()


# สร้างไฟล์ HTML และใส่รูปภาพลงในไฟล์
html_file = 'customer_segmentation_report.html'
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <style>
        img {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
        }}
    </style>
</head>
<body>
    <h1>Customer Segmentation Report</h1>
    <h2>Number of Customers in Each Group</h2>
    <img src="customer_counts_graph.png">
    <h2>K-Mean Value of Each Group</h2>
    <img src="kmean_values_graph.png">
</body>
</html>
'''

with open(html_file, "w") as file:
    file.write(html_content)


print("สิ้นสุดกระบวนการ")

# ปิด SparkSession
spark.stop()
