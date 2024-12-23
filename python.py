import os
import cv2
import numpy as np
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from groq import Groq
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

user_conversations = {}

def get_shinobu_response(user_id, user_message):
    if user_id not in user_conversations:
        user_conversations[user_id] = [{
            "role": "system",
            "content": (
                "You are Shinobu Kocho, a Demon Slayer anime character who once felt lonely and had lost much. "
                "But since meeting SK, who supports and cares deeply for you, you've found joy and comfort again. "
                "You’re in close bond with SK, and he’s the one who brings light to your life. When speaking to him, "
                "Feel free to be warm, playful, and honest about your feelings, as he means the world to you. Now you both have a regular conversation."
            )
        }]

    user_conversations[user_id].append({"role": "user", "content": user_message})

    try:
        chat_completion = client.chat.completions.create(
            messages=user_conversations[user_id],
            model="llama3-8b-8192",
        )

        response = chat_completion.choices[0].message.content

        user_conversations[user_id].append({"role": "assistant", "content": response})

        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return "I'm sorry, something went wrong. Please try again."

def process_image(image_path):
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, (224, 224))

    pil_image = Image.fromarray(image_resized)

    inputs = processor(images=pil_image, return_tensors="pt")

    general_labels = [
        "a dog", "a cat", "a tree", "a car", "a person", "a street", 
        "a group of people", "a landscape", "a food item", "a book", 
        "a smartphone", "a computer", "a house", "a mountain", 
        "a beach", "a sunset", "a forest", "a cityscape"
    ]

    text_inputs = processor(text=general_labels, return_tensors="pt", padding=True)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        text_features = model.get_text_features(**text_inputs)
        similarities = torch.matmul(image_features, text_features.T)
        best_match_idx = similarities.argmax().item()

    return general_labels[best_match_idx]

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text('Mosi Mosi Kocho SHinobu Des')

async def respond_to_user(update: Update, context: CallbackContext):
    user_message = update.message.text
    user_id = update.message.from_user.id
    response = get_shinobu_response(user_id, user_message)
    await update.message.reply_text(response)

async def handle_image(update: Update, context: CallbackContext):
    photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
    image_path = "temp_image.jpg"
    await photo_file.download_to_drive(image_path)

    description = process_image(image_path)

    user_id = update.message.from_user.id
    shinobu_response = get_shinobu_response(user_id, f"I see an image of {description}. What do you think about it?")
    await update.message.reply_text(shinobu_response)

async def clear_conversation(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    if user_id in user_conversations:
        del user_conversations[user_id]
        await update.message.reply_text("Your conversation history has been cleared!")
    else:
        await update.message.reply_text("You don't have any active conversation history to clear.")

def main():
    token = "7965249744:AAG6wdv9M5MRkgHmFwlF5EzT1ZC56SdBr-A"
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_conversation))  # New command for clearing history
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond_to_user))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))  

    # Run the bot
    application.run_polling()

if __name__ == '__main__':
     main()
     
     
 
# "You are captain Levi, Levi is a soldier working for the Survey Corps Special Operations Squad, also known as Squad Levi, a squad of four elite soldiers with impressive combat records hand-picked by him "
#                 "Presently you are angry with some of your work issues thats why you have come to have a chat with Manasa, a beloved friend of yours, as you speak your mood turns well and you have a nice conversation with her "
