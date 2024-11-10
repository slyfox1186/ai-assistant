# train_calculator.py

class TrainCalculator:
    def __init__(self, distance=None, speed=None):
        """Initialize calculator with optional distance and speed."""
        try:
            self.distance = float(distance) if distance is not None else None  # in miles
            self.speed = float(speed) if speed is not None else None      # in mph
        except (ValueError, TypeError):
            self.distance = None
            self.speed = None
        
    def calculate_time_minutes(self) -> float:
        """Calculate journey time in minutes."""
        try:
            if not self.distance or not self.speed:
                return None
            hours = self.distance / self.speed
            minutes = hours * 60
            return round(minutes)
        except Exception as e:
            print(f"Error calculating minutes: {e}")
            return None
        
    def calculate_time_hours(self) -> float:
        """Calculate journey time in hours (with decimals)."""
        try:
            if not self.distance or not self.speed:
                return None
            return self.distance / self.speed
        except Exception as e:
            print(f"Error calculating hours: {e}")
            return None
        
    def get_formatted_result(self) -> str:
        """Get a nicely formatted result with both hours and minutes."""
        try:
            if not self.distance or not self.speed:
                return "Please provide both distance and speed values."
                
            minutes = self.calculate_time_minutes()
            hours = self.calculate_time_hours()
            
            if minutes is None or hours is None:
                return "Error calculating journey time."
            
            return (f"Train Journey Details:\n"
                    f"Distance: {self.distance} miles\n"
                    f"Speed: {self.speed} mph\n"
                    f"Time: {hours:.2f} hours\n"
                    f"Time: {minutes} minutes")
        except Exception as e:
            print(f"Error formatting result: {e}")
            return "Error generating journey details."
                
    def parse_question(self, text: str) -> bool:
        """Parse a natural language question about train journey time."""
        try:
            if not text or not isinstance(text, str):
                return False
                
            import re
            
            # Look for distance and speed in the text
            distance_pattern = r"(\d+(?:\.\d+)?)\s*miles?"
            speed_pattern = r"(\d+(?:\.\d+)?)\s*mph"
            
            distance_match = re.search(distance_pattern, text.lower())
            speed_match = re.search(speed_pattern, text.lower())
            
            if distance_match and speed_match:
                try:
                    self.distance = float(distance_match.group(1))
                    self.speed = float(speed_match.group(1))
                    return True
                except ValueError:
                    print("Error converting values to float")
                    return False
            return False
            
        except Exception as e:
            print(f"Error parsing question: {e}")
            return False

# Example usage
if __name__ == "__main__":
    try:
        # Your specific problem
        train = TrainCalculator(distance=510, speed=63)
        print(train.get_formatted_result())
        
        # Allow for user input
        print("\nCalculate another journey:")
        try:
            dist = float(input("Enter distance (miles): "))
            spd = float(input("Enter speed (mph): "))
            
            new_train = TrainCalculator(dist, spd)
            print("\n" + new_train.get_formatted_result())
        except ValueError:
            print("Please enter valid numbers for distance and speed.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
    except Exception as e:
        print(f"Fatal error: {e}")