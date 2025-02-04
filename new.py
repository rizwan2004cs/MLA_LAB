import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

emails = [
    "Congratulations! You've won a free iPhone! Call now at 123456789",
    "Hi, are we still on for the meeting tomorrow at 10 AM?",
    "Limited time offer: Get a free trial of our premium software now!",
    "Hey, how's the new project going? Let me know if you need any help.",
    "Exclusive Deal! Buy one get one free on all gadgets. Hurry up!",
    "Reminder: Your Amazon account is about to expire. Please update your payment info.",
    "Good morning, just wanted to check in and see how you're doing!",
    "Urgent: Your bank account has been compromised. Please reset your password immediately.",
    "Win $1000 cash instantly! Call now to claim your prize!",
    "Let's catch up soon over coffee, it's been too long!",
    "Get paid to take surveys online! Sign up now to start earning.",
    "Hi, I hope you're doing well! I wanted to follow up on our last conversation.",
    "Huge discount on electronics! Shop now and save up to 50%!",
    "Special offer: Save 25% on all home appliances today only!",
    "Hey! Let's get together for a quick dinner this weekend!",
    "Claim your free vacation now! All expenses covered, just book your flight.",
    "Get your free credit report! Check for errors and improve your score.",
    "Only a few spots left for our summer sale. Don't miss out!",
    "The latest fashion trends at unbeatable prices. Shop now!",
    "You’ve been selected for a free trial of our fitness app. Start your journey today!"
]

labels = [
    1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0
]

# Vectorizing the emails
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)

# Training the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Generating a classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the Confusion Matrix using matplotlib and seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualizing class distribution
plt.figure(figsize=(6, 5))
plt.bar(["Ham", "Spam"], [labels.count(0), labels.count(1)], color=['green', 'red'])
plt.title('Class Distribution in the Dataset')
plt.xlabel('Class')
plt.ylabel('Number of Emails')
plt.show()

