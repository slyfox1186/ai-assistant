#!/usr/bin/env python3

import random
from database_handler import DatabaseHandler
from collections import namedtuple

# Define named tuples for structured data
President = namedtuple('President', ['name', 'party', 'term_start', 'term_end'])
State = namedtuple('State', ['name', 'capital', 'population'])
PoliticalParty = namedtuple('PoliticalParty', ['name', 'founded', 'ideology'])

class PresidentialTrainer:
    def __init__(self, db_name='ai_assistant.db'):
        self.db_handler = DatabaseHandler(db_name)

    def train_database(self):
        training_data = self.generate_training_data()
        for category, data in training_data.items():
            self.db_handler.store_training_data(category, data)
        print("Training data stored in the database.")

    def generate_training_data(self):
        training_data = {
            'mens_names': self.generate_mens_names(),
            'us_presidents': self.generate_us_presidents(),
            'political_parties': self.generate_political_parties(),
            'us_states': self.generate_us_states(),
            'government_positions': self.generate_government_positions(),
            'election_years': self.generate_election_years(),
            'presidential_facts': self.generate_presidential_facts(),
            'historical_events': self.generate_historical_events(),
            'constitutional_amendments': self.generate_constitutional_amendments(),
            'presidential_powers': self.generate_presidential_powers(),
            'white_house_rooms': self.generate_white_house_rooms(),
            'presidential_landmarks': self.generate_presidential_landmarks(),
            'first_ladies': self.generate_first_ladies(),
            'cabinet_positions': self.generate_cabinet_positions(),
            'presidential_quotes': self.generate_presidential_quotes(),
            'inauguration_dates': self.generate_inauguration_dates(),
            'presidential_libraries': self.generate_presidential_libraries(),
            'vice_presidents': self.generate_vice_presidents(),
            'presidential_nicknames': self.generate_presidential_nicknames(),
            'executive_orders': self.generate_executive_orders()
        }
        return training_data

    def generate_mens_names(self):
        return tuple([
            "John", "William", "James", "George", "Joseph", "Michael", "Thomas", "David",
            "Charles", "Robert", "Richard", "Donald", "Ronald", "Barack", "Abraham",
            "Theodore", "Franklin", "Dwight", "Lyndon", "Gerald", "Jimmy", "Bill", "Harry"
        ])

    def generate_us_presidents(self):
        return tuple([
            President("Joe Biden", "Democratic", 2021, None),
            President("Donald Trump", "Republican", 2017, 2021),
            President("Barack Obama", "Democratic", 2009, 2017),
            President("George W. Bush", "Republican", 2001, 2009),
            President("Bill Clinton", "Democratic", 1993, 2001)
        ])

    def generate_political_parties(self):
        return tuple([
            PoliticalParty("Democratic Party", 1828, "Center-left"),
            PoliticalParty("Republican Party", 1854, "Center-right"),
            PoliticalParty("Libertarian Party", 1971, "Right-libertarianism"),
            PoliticalParty("Green Party", 1984, "Green politics")
        ])

    def generate_us_states(self):
        return tuple([
            State("California", "Sacramento", 39538223),
            State("Texas", "Austin", 29145505),
            State("Florida", "Tallahassee", 21538187),
            State("New York", "Albany", 20201249),
            State("Pennsylvania", "Harrisburg", 13002700)
        ])

    def generate_government_positions(self):
        return tuple([
            "President", "Vice President", "Secretary of State", "Secretary of the Treasury",
            "Secretary of Defense", "Attorney General", "Speaker of the House",
            "Senate Majority Leader", "Chief Justice of the Supreme Court"
        ])

    def generate_election_years(self):
        return tuple(range(2024, 1788, -4))

    def generate_presidential_facts(self):
        return tuple([
            "The president must be at least 35 years old",
            "The president can serve a maximum of two four-year terms",
            "The president is also the Commander-in-Chief of the armed forces",
            "The president has the power to veto legislation",
            "The president can grant pardons for federal crimes"
        ])

    def generate_historical_events(self):
        return tuple([
            "Declaration of Independence (1776)",
            "Constitution ratified (1788)",
            "Louisiana Purchase (1803)",
            "Civil War (1861-1865)",
            "World War II (1941-1945)"
        ])

    def generate_constitutional_amendments(self):
        return tuple([
            "First Amendment: Freedom of speech, religion, and the press",
            "Second Amendment: Right to bear arms",
            "Thirteenth Amendment: Abolition of slavery",
            "Nineteenth Amendment: Women's right to vote",
            "Twenty-Second Amendment: Presidential term limits"
        ])

    def generate_presidential_powers(self):
        return tuple([
            "Veto legislation", "Grant pardons", "Nominate federal judges",
            "Make treaties", "Appoint cabinet members", "Command the military"
        ])

    def generate_white_house_rooms(self):
        return tuple([
            "Oval Office", "East Room", "Blue Room", "Green Room", "Red Room",
            "State Dining Room", "Situation Room", "Lincoln Bedroom"
        ])

    def generate_presidential_landmarks(self):
        return tuple([
            "Mount Rushmore", "Washington Monument", "Lincoln Memorial",
            "Jefferson Memorial", "Theodore Roosevelt Island"
        ])

    def generate_first_ladies(self):
        return tuple([
            "Jill Biden", "Melania Trump", "Michelle Obama",
            "Laura Bush", "Hillary Clinton", "Barbara Bush"
        ])

    def generate_cabinet_positions(self):
        return tuple([
            "Secretary of State", "Secretary of the Treasury", "Secretary of Defense",
            "Attorney General", "Secretary of the Interior", "Secretary of Agriculture",
            "Secretary of Commerce", "Secretary of Labor", "Secretary of Health and Human Services"
        ])

    def generate_presidential_quotes(self):
        return tuple([
            "Ask not what your country can do for you – ask what you can do for your country. - John F. Kennedy",
            "The only thing we have to fear is fear itself. - Franklin D. Roosevelt",
            "Government of the people, by the people, for the people, shall not perish from the Earth. - Abraham Lincoln",
            "Speak softly and carry a big stick. - Theodore Roosevelt",
            "Yes we can. - Barack Obama"
        ])

    def generate_inauguration_dates(self):
        return tuple([
            "January 20, 2021 - Joe Biden",
            "January 20, 2017 - Donald Trump",
            "January 20, 2009 - Barack Obama",
            "January 20, 2001 - George W. Bush",
            "January 20, 1993 - Bill Clinton"
        ])

    def generate_presidential_libraries(self):
        return tuple([
            "Barack Obama Presidential Center (planned)",
            "George W. Bush Presidential Library",
            "William J. Clinton Presidential Library",
            "George H.W. Bush Presidential Library",
            "Ronald Reagan Presidential Library"
        ])

    def generate_vice_presidents(self):
        return tuple([
            "Kamala Harris", "Mike Pence", "Joe Biden", "Dick Cheney", "Al Gore"
        ])

    def generate_presidential_nicknames(self):
        return tuple([
            "Honest Abe - Abraham Lincoln",
            "FDR - Franklin D. Roosevelt",
            "Ike - Dwight D. Eisenhower",
            "JFK - John F. Kennedy",
            "The Gipper - Ronald Reagan"
        ])

    def generate_executive_orders(self):
        return tuple([
            "Emancipation Proclamation - Abraham Lincoln",
            "Executive Order 9066 (Japanese internment) - Franklin D. Roosevelt",
            "Executive Order 10730 (Desegregation of schools) - Dwight D. Eisenhower",
            "Executive Order 11246 (Equal Employment Opportunity) - Lyndon B. Johnson",
            "Executive Order 13769 (Travel ban) - Donald Trump"
        ])

    def train_database(self):
        training_data = self.generate_training_data()
        for category, data in training_data.items():
            self.db_handler.store_training_data(category, data)
        print("Training data stored in the database.")

    def print_training_data(self):
        training_data = self.generate_training_data()
        for category, data in training_data.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for item in data[:5]:  # Print first 5 items of each category
                print(f"  - {item}")
            if len(data) > 5:
                print(f"  ... ({len(data) - 5} more items)")

if __name__ == "__main__":
    trainer = PresidentialTrainer()
    trainer.train_database()
    trainer.print_training_data()