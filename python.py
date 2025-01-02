import os
import cv2
import numpy as np
import requests
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from groq import Groq
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


STABILITY_API_KEY = "sk-pEZiZvrSENAoGOiucUN8r84Z5cFpF048elXtE5YDwkp8y54S"


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


def generate_image_stability(prompt: str) -> str:
    url = "https://api.stability.ai/v1/generation/stable-diffusion-v2-1-768-v2-1-2/model"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}"}
    data = {
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "samples": 1,
        "seed": 12345
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        image_url = response.json()["artifacts"][0]["image_url"]
        return image_url
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Download image from URL
def download_image(image_url: str, file_path: str):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Error downloading image: {response.status_code}, {response.text}")

# Command to start the bot
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text('Mosi Mosi Kocho Shinobu Des')

# Respond to user messages
async def respond_to_user(update: Update, context: CallbackContext):
    if update.message and update.message.text:
        user_message = update.message.text
        user_id = update.message.from_user.id
        response = get_shinobu_response(user_id, user_message)
        await update.message.reply_text(response)
    else:
        await update.message.reply_text("Sorry, I can only respond to text messages.")


async def handle_image(update: Update, context: CallbackContext):
    if update.message.photo:
        photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
        image_path = "temp_image.jpg"
        await photo_file.download_to_drive(image_path)

        description = process_image(image_path)

        user_id = update.message.from_user.id
        shinobu_response = get_shinobu_response(user_id, f"I see an image of {description}. What do you think about it?")
        await update.message.reply_text(shinobu_response)


async def generate_image(update: Update, context: CallbackContext):
    if update.message:  # Ensure update.message is not None
        if context.args:
            prompt = " ".join(context.args)
            await update.message.reply_text(f"Generating image for: {prompt}...")

            image_url = generate_image_stability(prompt)

            if image_url:

                image_path = "generated_image.jpg"
                download_image(image_url, image_path)


                await update.message.reply_text(f"Here is your generated image for: {prompt}")
                with open(image_path, "rb") as image_file:
                    await update.message.reply_photo(image_file)
            else:
                await update.message.reply_text("Sorry, something went wrong while generating the image.")
        else:
            await update.message.reply_text("Please provide a description for the image. Usage: /generate_image <description>")
    else:
        await update.message.reply_text("Sorry, I couldn't process your request.")


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
    application.add_handler(CommandHandler("clear", clear_conversation))
    application.add_handler(CommandHandler("generate_image", generate_image))  # Added new command for image generation
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond_to_user))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))


    application.run_polling()

if __name__ == '__main__':
    main()
