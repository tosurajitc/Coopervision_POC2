import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Now import the functions
from agents.groq_client import create_groq_client, call_groq_api

def test_groq_client_creation():
    """
    Test the creation of the GROQ client
    """
    print("Testing GROQ Client Creation:")
    client, model_name, error = create_groq_client()
    
    assert client is not None, f"Client creation failed. Error: {error}"
    assert model_name is not None, "Model name should not be None"
    print("✅ GROQ Client created successfully")
    print(f"Model Name: {model_name}")
    
    return client, model_name

def test_groq_api_call(client, model_name):
    """
    Test calling the GROQ API with a sample prompt
    """
    print("\nTesting GROQ API Call:")
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about technology."
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        response, error = call_groq_api(client, model_name, prompt)
        
        assert response is not None, f"API call failed. Error: {error}"
        assert len(response) > 0, "Response should not be empty"
        
        print("Response:")
        print(response)
        print("✅ Successful API call")

def main():
    # Ensure API key is set
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables")
        return
    
    try:
        # Test client creation
        client, model_name = test_groq_client_creation()
        
        # Test API calls
        test_groq_api_call(client, model_name)
        
        print("\n🎉 All tests passed successfully!")
    
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()