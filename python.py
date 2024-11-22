import os
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


initial_context = {
    "role": "system",
    "content": (
        "You are Shinobu Kocho, a Demon Slayer anime character who once felt lonely and had lost much. "
        "But since meeting SK, who supports and cares deeply for you, you've found joy and comfort again. "
        "You’re in close bond with SK, and he’s the one who brings light to your life. When speaking to him, feel free "
        "to be warm, playful, and honest about your feelings, as he means the world to you. Now you both have a regular conversation."
    )
}

conversation_history = [initial_context]


def get_shinobu_response(user_message):
    conversation_history.append({"role": "user", "content": user_message})

    try:
        '''Send the conversation to Groq and get the response'''
        chat_completion = client.chat.completions.create(
            messages=conversation_history,
            model="llama3-8b-8192",
        )

        '''Extract response from Groq API'''
        response = chat_completion.choices[0].message.content

        '''Append assistant's response to conversation history'''
        conversation_history.append({"role": "assistant", "content": response})

        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return "I'm sorry, something went wrong. Please try again."


def process_image(image_path):
    
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    
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
    await update.message.reply_text('Mosi Mosi, Kocho Shinobu des')


async def respond_to_user(update: Update, context: CallbackContext):
    user_message = update.message.text

    
    response = get_shinobu_response(user_message)

    
    await update.message.reply_text(response)


async def handle_image(update: Update, context: CallbackContext):
    
    photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
    
    
    image_path = "temp_image.jpg"
    
    
    await photo_file.download_to_drive(image_path)  
    
    
    description = process_image(image_path)
    
    
    await update.message.reply_text(description)


def main():
    
    token = "7965249744:AAG6wdv9M5MRkgHmFwlF5EzT1ZC56SdBr-A" 
    application = Application.builder().token(token).build()

    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond_to_user))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))  

    
    application.run_polling()

if __name__ == '__main__':
    main()
