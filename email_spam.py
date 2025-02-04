
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

#This the dataset that is used for both training and testing
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
    "Special offer: Save 25percent on all home appliances today only!",
    "Hey! Let's get together for a quick dinner this weekend!",
    "Claim your free vacation now! All expenses covered, just book your flight.",
    "Get your free credit report! Check for errors and improve your score.",
    "Only a few spots left for our summer sale. Don't miss out!",
    "The latest fashion trends at unbeatable prices. Shop now!",
    "You've been selected for a free trial of our fitness app. Start your journey today!"
]

# This is the label for the above dataset ( 1 = spam ,0 = ham)
labels = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0]

#This creates a word list matrix for counting each word
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

#This is used to split the dataset for training ad testing
x_train,x_test,y_train,y_test = train_test_split(x,labels,test_size=0.2,random_state=42)

#The MultinomialNB is an model trainer we train the model with datset and label 
model = MultinomialNB()
model.fit(x_train,y_train)

#We predict the test dataset using the above model using model created using naive_bayes
y_pred = model.predict(x_test)

#We are comparing the predicted result with actual labels for accuracy
print(classification_report(y_pred,y_test))


