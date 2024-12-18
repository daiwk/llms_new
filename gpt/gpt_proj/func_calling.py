# Simulate the order_id and delivery_date
from datetime import datetime
from openai import OpenAI
import json

client = OpenAI()
order_id = "order_12345"

def get_delivery_date(order_id):
    delivery_date = datetime.now()
    return delivery_date



tools = [
  {
      "type": "function",
      "function": {
          "name": "get_delivery_date",
          "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
          "parameters": {
              "type": "object",
              "properties": {
                  "order_id": {
                      "type": "string",
                      "description": "The customer's order ID."
                  }
              },
              "required": ["order_id"],
              "additionalProperties": False
          }
      }
  }
]

messages = []
messages.append({"role": "system", "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."})
messages.append({"role": "user", "content": "Hi, can you tell me the delivery date for my order?"})
messages.append({"role": "assistant", "content": "Hi there! I can help with that. Can you please provide your order ID?"})
messages.append({"role": "user", "content": "i think it is order_12345"})

response = client.chat.completions.create(
  model='gpt-4o',
  messages=messages,
  tools=tools
)


### Simulate the tool call response
##
##response = {
##  "choices": [
##      {
##          "message": {
##              "role": "assistant",
##              "tool_calls": [
##                  {
##                      "id": "call_62136354",
##                      "type": "function",
##                      "function": {
##                          "arguments": "{'order_id': 'order_12345'}",
##                          "name": "get_delivery_date"
##                      }
##                  }
##              ]
##          }
##      }
##  ]
##}

print(response, "mmmm")
tool_call = response.choices[0].message.tool_calls[0]
print(tool_call, 'jjjj')
#arguments = json.loads(tool_call['function']['arguments'])
arguments = json.loads(tool_call.function.arguments)

order_id = arguments.get('order_id')
print(order_id, "dwk")
delivery_date = get_delivery_date(order_id) # call real function

# Create a message containing the result of the function call

function_call_result_message = {
  "role": "tool",
  "content": json.dumps({
      "order_id": order_id,
      "delivery_date": delivery_date.strftime('%Y-%m-%d %H:%M:%S')
  }),
  "tool_call_id": response.choices[0].message.tool_calls[0].id
}

# Prepare the chat completion call payload

completion_payload = {
  "model": "gpt-4o",
  "messages": [
      {"role": "system", "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."},
      {"role": "user", "content": "Hi, can you tell me the delivery date for my order?"},
      {"role": "assistant", "content": "Hi there! I can help with that. Can you please provide your order ID?"},
      {"role": "user", "content": "i think it is order_12345"},
      response.choices[0].message,
      function_call_result_message
  ]
}

# Call the OpenAI API's chat completions endpoint to send the tool call result back to the model

response = client.chat.completions.create(
  model=completion_payload["model"],
  messages=completion_payload["messages"]
)

# Print the response from the API. In this case it will typically contain a message such as "The delivery date for your order #12345 is xyz. Is there anything else I can help you with?"

print(response)
