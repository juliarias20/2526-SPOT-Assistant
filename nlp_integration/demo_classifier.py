from semantic_task_classifier import SemanticTaskClassifier

def main():
    parser = SemanticTaskClassifier()
    print("\n== SPOT Assistant NLP Test ===")
    print("Type a command (e.g., 'Bring me a notebook.') or 'q' to quit.\n")

    while True:
        user_input = input("Command: ").strip()
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Exiting NLP test.")
            break
        result = parser.parse_command(user_input)
        print(f"\nParsed Output: \n{result}\n")

if __name__ == "__main__":
    main()