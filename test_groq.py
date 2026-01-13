from groq import Groq

client = Groq(api_key="Your_API_KEY")

completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello, Groq!"}],
)

print(completion.choices[0].message.content)
