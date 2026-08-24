from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ข้อมูลฝึกเบื้องต้น
texts = [
    "เครียดมาก",
    "เศร้าจัง",
    "อยากตาย",
    "เหนื่อยไม่ไหว",
    "มีความสุข",
    "สบายดี",
    "วันนี้ดีมาก"
]

labels = [
    "เครียด",
    "ซึมเศร้า",
    "เสี่ยงสูง",
    "เครียด",
    "ปกติ",
    "ปกติ",
    "ปกติ"
]

# แปลงข้อความเป็นตัวเลข
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# เทรน SVM
model = LinearSVC()
model.fit(X, labels)

# ทดสอบ
test = ["เหนื่อยมาก ไม่อยากทำอะไร"]
X_test = vectorizer.transform(test)

result = model.predict(X_test)

print("ผลลัพธ์ =", result[0])