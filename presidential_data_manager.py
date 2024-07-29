#!/usr/bin/env python3

import json
import os

class PresidentialDataManager:
    def __init__(self, data_directory='training_data'):
        self.data_directory = data_directory
        self.data = self.load_all_data()

    def load_all_data(self):
        data = {}
        for filename in os.listdir(self.data_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_directory, filename)
                with open(file_path, 'r') as file:
                    data[filename] = json.load(file)
        return data

    def display_data(self):
        for filename, content in self.data.items():
            print(f"\nData from {filename}:")
            print(json.dumps(content, indent=2))

    def add_data(self, category, item):
        for filename, content in self.data.items():
            if category in content:
                content[category].append(item)
                self.save_data(filename, content)
                print(f"Added {item} to {category} in {filename}")

    def remove_data(self, category, item):
        for filename, content in self.data.items():
            if category in content and item in content[category]:
                content[category].remove(item)
                self.save_data(filename, content)
                print(f"Removed {item} from {category} in {filename}")

    def save_data(self, filename, content):
        file_path = os.path.join(self.data_directory, filename)
        with open(file_path, 'w') as file:
            json.dump(content, file, indent=2)

    def run_shell(self):
        while True:
            command = input("\nEnter a command (add, remove, display, train, quit): ").strip().lower()
            if command == 'quit':
                break
            elif command == 'display':
                self.display_data()
            elif command.startswith('add'):
                _, category, item = command.split(maxsplit=2)
                self.add_data(category, item)
            elif command.startswith('remove'):
                _, category, item = command.split(maxsplit=2)
                self.remove_data(category, item)
            elif command == 'train':
                print("Training data added to the training database.")
            else:
                print("Unknown command. Please try again.")

if __name__ == "__main__":
    manager = PresidentialDataManager()
    manager.run_shell()
