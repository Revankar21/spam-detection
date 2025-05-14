import pandas as pd
import numpy as np
from datetime import datetime
import random

# Lists of common words and phrases for generating synthetic data
spam_subjects = [
    "CONGRATULATIONS", "WIN", "FREE", "DISCOUNT", "LIMITED TIME", "EXCLUSIVE OFFER",
    "URGENT", "MONEY", "GUARANTEED", "INSTANT", "CASH", "PRIZE", "WINNER",
    "INVESTMENT", "OPPORTUNITY", "LUXURY", "CHEAP", "BUY NOW", "SALE"
]

spam_bodies = [
    "You've won {amount}!", 
    "Claim your free {item} now!",
    "Limited time offer: {discount}% off!",
    "Congratulations! You've been selected for {prize}",
    "URGENT: Your account needs verification",
    "Make {amount} working from home!",
    "Investment opportunity: Guaranteed {return_rate}% returns",
    "Buy authentic {brand} watches at {discount}% off",
    "Your lottery winning of {amount} is pending",
    "Exclusive access to {product} at wholesale prices"
]

ham_subjects = [
    "Meeting", "Update", "Question", "Hello", "Information", "Request",
    "Follow-up", "Reminder", "Schedule", "Project", "Report", "Thanks",
    "Important", "Review", "Discussion", "Feedback", "Notice", "Team"
]

ham_bodies = [
    "Hi, could you please review the {document} when you have a chance?",
    "Just following up on our discussion about {topic}.",
    "The meeting is scheduled for {time}. Please confirm if you can attend.",
    "Thank you for your help with the {project}.",
    "Here are the updates for the {project} project.",
    "Please find attached the {document} for your review.",
    "Can we schedule a meeting to discuss {topic}?",
    "Just wanted to remind you about the {event} tomorrow.",
    "Great work on the {project}! Let's discuss next steps.",
    "I've reviewed your proposal and have some feedback."
]

products = ["iPhone", "laptop", "watch", "tablet", "headphones", "camera", "TV", "console"]
brands = ["Rolex", "Nike", "Apple", "Samsung", "Sony", "LG", "Dell", "HP"]
amounts = ["$1,000", "$5,000", "$10,000", "$50,000", "$100,000", "$1,000,000"]
discounts = [50, 70, 80, 90, 95]
documents = ["report", "presentation", "proposal", "document", "spreadsheet"]
projects = ["marketing", "sales", "development", "research", "design"]
topics = ["budget", "timeline", "requirements", "strategy", "goals"]
times = ["2 PM", "3:30 PM", "10 AM", "11:15 AM", "4 PM"]
events = ["meeting", "presentation", "workshop", "conference", "training"]

def generate_spam_email():
    subject = random.choice(spam_subjects)
    template = random.choice(spam_bodies)
    
    # Fill in template with random values
    email = template.format(
        amount=random.choice(amounts),
        item=random.choice(products),
        discount=random.choice(discounts),
        prize=random.choice(products),
        return_rate=random.randint(100, 1000),
        brand=random.choice(brands),
        product=random.choice(products)
    )
    
    # Add some random capitalization and special characters for spam-like appearance
    if random.random() < 0.3:
        email = email.upper()
    if random.random() < 0.4:
        email = email.replace('o', '0').replace('i', '1').replace('s', '$')
    
    return "spam", email

def generate_ham_email():
    subject = random.choice(ham_subjects)
    template = random.choice(ham_bodies)
    
    # Fill in template with random values
    email = template.format(
        document=random.choice(documents),
        topic=random.choice(topics),
        time=random.choice(times),
        project=random.choice(projects),
        event=random.choice(events)
    )
    
    return "ham", email

# Generate synthetic data
synthetic_data = []
for _ in range(2500):  # Generate 2500 spam emails
    label, text = generate_spam_email()
    synthetic_data.append({'v1': label, 'v2': text})

for _ in range(2500):  # Generate 2500 ham emails
    label, text = generate_ham_email()
    synthetic_data.append({'v1': label, 'v2': text})

# Create DataFrame with synthetic data
synthetic_df = pd.DataFrame(synthetic_data)

# Read existing data
try:
    existing_df = pd.read_csv('spam.csv', encoding='latin-1')
    # Combine existing and synthetic data
    combined_df = pd.concat([existing_df, synthetic_df], ignore_index=True)
except FileNotFoundError:
    combined_df = synthetic_df

# Save combined dataset
combined_df.to_csv('spam.csv', index=False, encoding='latin-1')

print(f"Total number of emails in dataset: {len(combined_df)}")
print(f"Number of spam emails: {len(combined_df[combined_df['v1'] == 'spam'])}")
print(f"Number of ham emails: {len(combined_df[combined_df['v1'] == 'ham'])}") 