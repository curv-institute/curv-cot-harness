#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Generate semantic task datasets for Experiment 005.

Creates 4 families:
- commonsense: everyday physical/social reasoning
- causal: multi-step story reasoning
- distractor: word problems with irrelevant details
- explanation: hypothesis selection tasks

Each family has 30 items with gold answers and plausible alternatives.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Family A: Commonsense QA
COMMONSENSE_ITEMS = [
    {
        "id": "cs-001",
        "prompt": "Sarah put a glass of water in the freezer before going to bed. What will she find in the morning?",
        "answer": "ice",
        "alternatives": ["cold water", "evaporated water"],
        "answer_type": "string"
    },
    {
        "id": "cs-002",
        "prompt": "A person wearing a heavy winter coat enters a warm building. What will they likely do?",
        "answer": "take off the coat",
        "alternatives": ["keep the coat on", "put on more layers"],
        "answer_type": "string"
    },
    {
        "id": "cs-003",
        "prompt": "Tom sees dark clouds gathering and hears thunder in the distance. What should he bring if going outside?",
        "answer": "umbrella",
        "alternatives": ["sunglasses", "sunscreen"],
        "answer_type": "string"
    },
    {
        "id": "cs-004",
        "prompt": "A cat is staring intensely at a bird outside the window. What is the cat most likely feeling?",
        "answer": "predatory interest or hunting instinct",
        "alternatives": ["fear", "indifference"],
        "answer_type": "string"
    },
    {
        "id": "cs-005",
        "prompt": "Maria notices her plant's leaves are turning yellow and drooping. The soil is very dry. What does the plant need?",
        "answer": "water",
        "alternatives": ["more sunlight", "fertilizer"],
        "answer_type": "string"
    },
    {
        "id": "cs-006",
        "prompt": "A child drops their ice cream cone on a hot summer sidewalk. What will happen to the ice cream?",
        "answer": "it will melt",
        "alternatives": ["it will stay frozen", "it will evaporate instantly"],
        "answer_type": "string"
    },
    {
        "id": "cs-007",
        "prompt": "John forgot to charge his phone overnight. When he wakes up, he finds his phone is dead. Why?",
        "answer": "the battery drained completely",
        "alternatives": ["the phone broke", "someone stole the battery"],
        "answer_type": "string"
    },
    {
        "id": "cs-008",
        "prompt": "A person at a restaurant waves at the waiter and points to their empty glass. What do they want?",
        "answer": "a refill of their drink",
        "alternatives": ["the check", "to complain about the glass"],
        "answer_type": "string"
    },
    {
        "id": "cs-009",
        "prompt": "Emma's friend hasn't responded to her texts for three days, which is unusual. What should Emma assume first?",
        "answer": "her friend might be busy or their phone has issues",
        "alternatives": ["her friend is angry at her", "her friend lost their phone permanently"],
        "answer_type": "string"
    },
    {
        "id": "cs-010",
        "prompt": "A rubber ball and a glass ball are dropped from the same height onto concrete. Which one is more likely to break?",
        "answer": "the glass ball",
        "alternatives": ["the rubber ball", "both will break equally"],
        "answer_type": "string"
    },
    {
        "id": "cs-011",
        "prompt": "David puts a pot of water on a hot stove and leaves it for 20 minutes. What will he find?",
        "answer": "boiling water or steam (water boiled off)",
        "alternatives": ["cold water", "frozen water"],
        "answer_type": "string"
    },
    {
        "id": "cs-012",
        "prompt": "A person yawns during a meeting. What does this most likely indicate?",
        "answer": "tiredness or boredom",
        "alternatives": ["excitement", "hunger"],
        "answer_type": "string"
    },
    {
        "id": "cs-013",
        "prompt": "Lisa leaves butter on the kitchen counter on a warm day. What will happen to it?",
        "answer": "it will soften or melt",
        "alternatives": ["it will harden", "it will stay the same"],
        "answer_type": "string"
    },
    {
        "id": "cs-014",
        "prompt": "A dog wags its tail rapidly when its owner comes home. What is the dog feeling?",
        "answer": "happiness or excitement",
        "alternatives": ["fear", "aggression"],
        "answer_type": "string"
    },
    {
        "id": "cs-015",
        "prompt": "Mike's car won't start and the dashboard lights are dim. What is the most likely problem?",
        "answer": "dead or low battery",
        "alternatives": ["flat tire", "empty windshield fluid"],
        "answer_type": "string"
    },
    {
        "id": "cs-016",
        "prompt": "A baby starts crying shortly after their last feeding. What might they need?",
        "answer": "a diaper change or comfort",
        "alternatives": ["more food immediately", "exercise"],
        "answer_type": "string"
    },
    {
        "id": "cs-017",
        "prompt": "Rachel sees her breath forming clouds in front of her face. What does this indicate about the temperature?",
        "answer": "it is cold outside",
        "alternatives": ["it is hot", "it is exactly room temperature"],
        "answer_type": "string"
    },
    {
        "id": "cs-018",
        "prompt": "A person's stomach growls loudly. What does this typically mean?",
        "answer": "they are hungry",
        "alternatives": ["they are full", "they are scared"],
        "answer_type": "string"
    },
    {
        "id": "cs-019",
        "prompt": "Kevin leaves his wet clothes outside on a sunny, windy day. What will happen to them?",
        "answer": "they will dry",
        "alternatives": ["they will get wetter", "they will freeze"],
        "answer_type": "string"
    },
    {
        "id": "cs-020",
        "prompt": "A person squints and shields their eyes while looking outside. What is the weather likely like?",
        "answer": "sunny or bright",
        "alternatives": ["dark and cloudy", "foggy"],
        "answer_type": "string"
    },
    {
        "id": "cs-021",
        "prompt": "Anna's houseplant in the closet has pale, stretched stems. What does it need?",
        "answer": "more light",
        "alternatives": ["more water", "colder temperature"],
        "answer_type": "string"
    },
    {
        "id": "cs-022",
        "prompt": "A child covers their ears during fireworks. Why?",
        "answer": "the noise is too loud",
        "alternatives": ["they want to hear better", "their ears are cold"],
        "answer_type": "string"
    },
    {
        "id": "cs-023",
        "prompt": "James notices his milk smells sour and has chunks. What should he do with it?",
        "answer": "throw it away, it has spoiled",
        "alternatives": ["drink it quickly", "heat it up to fix it"],
        "answer_type": "string"
    },
    {
        "id": "cs-024",
        "prompt": "A person at a party keeps checking their watch. What might they be thinking?",
        "answer": "they want to leave or have somewhere to be",
        "alternatives": ["they love the party", "they are admiring the watch"],
        "answer_type": "string"
    },
    {
        "id": "cs-025",
        "prompt": "Sophie's car makes a grinding noise when she brakes. What is likely wrong?",
        "answer": "brake pads are worn out",
        "alternatives": ["the radio is broken", "the gas tank is empty"],
        "answer_type": "string"
    },
    {
        "id": "cs-026",
        "prompt": "A balloon filled with helium is released indoors. Where will it go?",
        "answer": "up to the ceiling",
        "alternatives": ["straight down to the floor", "sideways to the wall"],
        "answer_type": "string"
    },
    {
        "id": "cs-027",
        "prompt": "Mark's flashlight beam gets dimmer over time during use. What is happening?",
        "answer": "the batteries are draining",
        "alternatives": ["the bulb is getting stronger", "the room is getting brighter"],
        "answer_type": "string"
    },
    {
        "id": "cs-028",
        "prompt": "A person laughs at something their friend whispers. What kind of thing was likely whispered?",
        "answer": "something funny or amusing",
        "alternatives": ["sad news", "important instructions"],
        "answer_type": "string"
    },
    {
        "id": "cs-029",
        "prompt": "Claire puts a metal spoon in a cup of hot soup. What will happen to the spoon?",
        "answer": "it will get hot",
        "alternatives": ["it will stay cold", "it will melt"],
        "answer_type": "string"
    },
    {
        "id": "cs-030",
        "prompt": "A person at a crosswalk presses the button and waits. What are they waiting for?",
        "answer": "the walk signal to cross the street",
        "alternatives": ["a bus", "the button to give them money"],
        "answer_type": "string"
    },
]

# Family B: Causal/Story Reasoning
CAUSAL_ITEMS = [
    {
        "id": "ca-001",
        "prompt": "Amy forgot to set her alarm. Her meeting was at 9 AM. She woke up naturally at 10 AM. What happened to her meeting?",
        "answer": "she missed it",
        "alternatives": ["she arrived early", "the meeting was rescheduled for her"],
        "answer_type": "string"
    },
    {
        "id": "ca-002",
        "prompt": "The road was icy. Jack was driving fast. He tried to brake suddenly. What likely happened next?",
        "answer": "the car skidded or slid",
        "alternatives": ["the car stopped instantly", "the ice melted"],
        "answer_type": "string"
    },
    {
        "id": "ca-003",
        "prompt": "Maria planted seeds but forgot to water them for a month. The weather was dry. What happened to the seeds?",
        "answer": "they did not grow or died",
        "alternatives": ["they grew into large plants", "they turned into flowers immediately"],
        "answer_type": "string"
    },
    {
        "id": "ca-004",
        "prompt": "Tom studied hard for months. He practiced past exams. He got enough sleep before the test. How did he likely perform?",
        "answer": "well or passed",
        "alternatives": ["he failed badly", "the test was cancelled because of him"],
        "answer_type": "string"
    },
    {
        "id": "ca-005",
        "prompt": "The restaurant was very popular. It was Saturday night. No reservation was made. What happened when they arrived?",
        "answer": "they had to wait or couldn't get a table",
        "alternatives": ["they got the best table immediately", "the restaurant closed for them"],
        "answer_type": "string"
    },
    {
        "id": "ca-006",
        "prompt": "The power went out during the storm. The freezer was full of food. The outage lasted three days. What happened to the food?",
        "answer": "it thawed and spoiled",
        "alternatives": ["it stayed frozen perfectly", "it cooked itself"],
        "answer_type": "string"
    },
    {
        "id": "ca-007",
        "prompt": "Sara left her chocolate bar in the car on a hot day. She returned after several hours. What did she find?",
        "answer": "melted chocolate",
        "alternatives": ["frozen chocolate", "the chocolate had multiplied"],
        "answer_type": "string"
    },
    {
        "id": "ca-008",
        "prompt": "The company's sales dropped. They reduced advertising to zero. They raised prices. What happened to sales next quarter?",
        "answer": "sales dropped further",
        "alternatives": ["sales increased dramatically", "competitors went out of business"],
        "answer_type": "string"
    },
    {
        "id": "ca-009",
        "prompt": "Ben didn't study any Spanish. He never practiced speaking. He took a Spanish proficiency test. How did he do?",
        "answer": "poorly or failed",
        "alternatives": ["he was fluent", "he became a translator"],
        "answer_type": "string"
    },
    {
        "id": "ca-010",
        "prompt": "The dam upstream was old and cracked. Heavy rains came for a week. The cracks widened. What happened to the town downstream?",
        "answer": "it flooded or was at risk of flooding",
        "alternatives": ["it experienced drought", "the dam fixed itself"],
        "answer_type": "string"
    },
    {
        "id": "ca-011",
        "prompt": "Jenny ate a large meal. Then she went swimming immediately. What did she likely experience?",
        "answer": "discomfort or cramps",
        "alternatives": ["increased energy", "hunger"],
        "answer_type": "string"
    },
    {
        "id": "ca-012",
        "prompt": "The factory released chemicals into the river. Fish began dying. The water changed color. What can we conclude about the river?",
        "answer": "it became polluted",
        "alternatives": ["it became cleaner", "the fish were unaffected"],
        "answer_type": "string"
    },
    {
        "id": "ca-013",
        "prompt": "Tim forgot his passport at home. He arrived at the airport. He was traveling internationally. What happened?",
        "answer": "he couldn't board the flight",
        "alternatives": ["he boarded without issue", "they gave him a new passport"],
        "answer_type": "string"
    },
    {
        "id": "ca-014",
        "prompt": "The orchestra hadn't rehearsed together. Each musician practiced different pieces. The concert was that night. How did it go?",
        "answer": "poorly or chaotic",
        "alternatives": ["perfectly", "they won an award"],
        "answer_type": "string"
    },
    {
        "id": "ca-015",
        "prompt": "Lisa didn't pay her electric bill for three months. The company sent warnings. She ignored them. What happened eventually?",
        "answer": "her electricity was disconnected",
        "alternatives": ["she got free electricity forever", "the bill disappeared"],
        "answer_type": "string"
    },
    {
        "id": "ca-016",
        "prompt": "The old bridge had warning signs. Max ignored them and drove across with a heavy truck. What likely happened?",
        "answer": "the bridge collapsed or was damaged",
        "alternatives": ["the bridge became stronger", "the truck flew over"],
        "answer_type": "string"
    },
    {
        "id": "ca-017",
        "prompt": "The team never practiced together. They didn't know each other's positions. They played against the champions. What was the result?",
        "answer": "they lost badly",
        "alternatives": ["they won the championship", "the game was cancelled in their honor"],
        "answer_type": "string"
    },
    {
        "id": "ca-018",
        "prompt": "Karen didn't water her lawn all summer. There was a drought. Fall arrived. What did her lawn look like?",
        "answer": "brown and dead",
        "alternatives": ["lush and green", "covered in flowers"],
        "answer_type": "string"
    },
    {
        "id": "ca-019",
        "prompt": "The scientist mixed the wrong chemicals. A reaction started. Smoke appeared. What should have happened next?",
        "answer": "evacuation or emergency response",
        "alternatives": ["a celebration", "the chemicals sorted themselves out"],
        "answer_type": "string"
    },
    {
        "id": "ca-020",
        "prompt": "Dan never exercised. He ate poorly for years. He avoided doctors. His health likely...",
        "answer": "deteriorated",
        "alternatives": ["improved dramatically", "stayed perfect"],
        "answer_type": "string"
    },
    {
        "id": "ca-021",
        "prompt": "The small business had no customers for months. The owner didn't change anything. Bills kept coming. What happened?",
        "answer": "it closed or went bankrupt",
        "alternatives": ["it became highly profitable", "customers appeared magically"],
        "answer_type": "string"
    },
    {
        "id": "ca-022",
        "prompt": "The weather forecast predicted a blizzard. Schools closed early. People stocked up on supplies. What was expected?",
        "answer": "heavy snow and dangerous conditions",
        "alternatives": ["a heat wave", "perfect weather"],
        "answer_type": "string"
    },
    {
        "id": "ca-023",
        "prompt": "Emily applied to 50 jobs. She tailored each application. She followed up professionally. What was the likely outcome?",
        "answer": "she got interview offers or a job",
        "alternatives": ["all 50 companies closed", "she gave up before hearing back"],
        "answer_type": "string"
    },
    {
        "id": "ca-024",
        "prompt": "The pipe had a small leak. No one fixed it. Winter came with freezing temperatures. What happened to the pipe?",
        "answer": "it burst or the leak worsened",
        "alternatives": ["it fixed itself", "the water inside stayed warm"],
        "answer_type": "string"
    },
    {
        "id": "ca-025",
        "prompt": "The chef didn't taste the soup. He added salt randomly. He served it to critics. How did they react?",
        "answer": "negatively, it was too salty or underseasoned",
        "alternatives": ["they gave it five stars", "they didn't notice any issues"],
        "answer_type": "string"
    },
    {
        "id": "ca-026",
        "prompt": "The student plagiarized their entire essay. The professor used plagiarism detection software. What was discovered?",
        "answer": "the plagiarism was caught",
        "alternatives": ["the essay was praised as original", "the software congratulated the student"],
        "answer_type": "string"
    },
    {
        "id": "ca-027",
        "prompt": "The fire alarm went off in the building. People smelled smoke. The exits were clearly marked. What should people have done?",
        "answer": "evacuated the building",
        "alternatives": ["ignored it and continued working", "started more fires"],
        "answer_type": "string"
    },
    {
        "id": "ca-028",
        "prompt": "Jake left his car unlocked with keys inside in a high-crime area overnight. What likely happened?",
        "answer": "the car was stolen or broken into",
        "alternatives": ["the car became more secure", "nothing, crime doesn't exist there"],
        "answer_type": "string"
    },
    {
        "id": "ca-029",
        "prompt": "The medication required refrigeration. It was left out for a week in summer heat. The patient took it. Was this safe?",
        "answer": "no, it may have degraded and become unsafe or ineffective",
        "alternatives": ["yes, completely safe", "the heat made it stronger"],
        "answer_type": "string"
    },
    {
        "id": "ca-030",
        "prompt": "The concert venue held 500 people. 1000 tickets were sold. Everyone showed up. What was the problem?",
        "answer": "overcrowding, not everyone could get in",
        "alternatives": ["the venue magically expanded", "half the people became invisible"],
        "answer_type": "string"
    },
]

# Family C: Distractor-heavy Word Problems
DISTRACTOR_ITEMS = [
    {
        "id": "di-001",
        "prompt": "A library has 5,000 books. The librarian's name is Margaret and she has worked there for 20 years. She wears glasses and has two cats. On Tuesday, 47 books were borrowed and 23 were returned. What is the librarian's name?",
        "answer": "Margaret",
        "alternatives": ["47", "20 years"],
        "answer_type": "string"
    },
    {
        "id": "di-002",
        "prompt": "Tom is taller than Jerry. Jerry is taller than Spike. Tom has brown hair. Jerry likes pizza. Spike weighs 150 pounds. Who is the shortest?",
        "answer": "Spike",
        "alternatives": ["Tom", "Jerry"],
        "answer_type": "string"
    },
    {
        "id": "di-003",
        "prompt": "A red car and a blue car are in a race. The red car is faster but started late. The blue car driver is named Sam who is 35 years old. The red car has racing stripes. Sam's favorite food is tacos. Which car is faster?",
        "answer": "the red car",
        "alternatives": ["the blue car", "Sam"],
        "answer_type": "string"
    },
    {
        "id": "di-004",
        "prompt": "Maria has three apples. Her sister Ana lives in Madrid and works as a teacher. Their mother's birthday is in June. Maria gave one apple to her friend. Their dog is named Bruno. How many apples does Maria have now?",
        "answer": "2",
        "alternatives": ["3", "June"],
        "answer_type": "string"
    },
    {
        "id": "di-005",
        "prompt": "In a classroom, there are 12 boys and 15 girls. The teacher Mr. Johnson has been teaching for 8 years. The classroom is on the third floor near the science lab. What is the total number of students?",
        "answer": "27",
        "alternatives": ["8", "third floor"],
        "answer_type": "string"
    },
    {
        "id": "di-006",
        "prompt": "A store sells apples for $2 each and oranges for $3 each. The store opened in 1995 and the owner drives a green truck. The store is open from 9am to 6pm. Which fruit is cheaper?",
        "answer": "apples",
        "alternatives": ["oranges", "1995"],
        "answer_type": "string"
    },
    {
        "id": "di-007",
        "prompt": "John can run faster than Mike. Mike can run faster than Steve. John's favorite color is blue. Mike was born in Chicago. Steve has a pet hamster named Fluffy. Who is the fastest runner?",
        "answer": "John",
        "alternatives": ["Mike", "Steve"],
        "answer_type": "string"
    },
    {
        "id": "di-008",
        "prompt": "The train leaves at 3:00 PM. The station was built in 1920 and has marble floors. The conductor's name is William. The train has 8 cars and travels to Boston. William has three children. What time does the train leave?",
        "answer": "3:00 PM",
        "alternatives": ["8", "1920"],
        "answer_type": "string"
    },
    {
        "id": "di-009",
        "prompt": "Lisa is older than Kate. Kate is older than Emma. Lisa enjoys painting. Kate plays tennis every Saturday. Emma's middle name is Rose. Who is the youngest?",
        "answer": "Emma",
        "alternatives": ["Lisa", "Kate"],
        "answer_type": "string"
    },
    {
        "id": "di-010",
        "prompt": "A farmer has chickens and cows. There are 10 chickens that each lay eggs daily. The farmer's name is Bob and he wears a straw hat. His farm is 50 acres and he wakes up at 5 AM. What animals produce the eggs?",
        "answer": "chickens",
        "alternatives": ["cows", "Bob"],
        "answer_type": "string"
    },
    {
        "id": "di-011",
        "prompt": "Box A weighs more than Box B. Box B weighs more than Box C. Box A is painted red. Box B contains books. Box C was shipped from Germany. Which box is lightest?",
        "answer": "Box C",
        "alternatives": ["Box A", "Box B"],
        "answer_type": "string"
    },
    {
        "id": "di-012",
        "prompt": "The movie starts at 7:30 PM and is 2 hours long. The theater has 300 seats and was renovated last year. Popcorn costs $8 and the director won an award. What time does the movie end?",
        "answer": "9:30 PM",
        "alternatives": ["7:30 PM", "$8"],
        "answer_type": "string"
    },
    {
        "id": "di-013",
        "prompt": "Sarah speaks French and Spanish. Her brother speaks only English. Sarah lives in an apartment on Oak Street. Her brother's apartment has a balcony. They both like coffee. How many languages does Sarah speak?",
        "answer": "2",
        "alternatives": ["1", "Oak Street"],
        "answer_type": "string"
    },
    {
        "id": "di-014",
        "prompt": "Plant A needs more water than Plant B. Plant B needs more water than Plant C. Plant A has red flowers. Plant B was bought at the garden center. Plant C is in a blue pot near the window. Which plant needs the least water?",
        "answer": "Plant C",
        "alternatives": ["Plant A", "Plant B"],
        "answer_type": "string"
    },
    {
        "id": "di-015",
        "prompt": "The bakery opens at 6 AM. The head baker Rosa has worked there for 15 years. They sell croissants for $3 each. The building was originally a pharmacy. Rosa's son helps on weekends. When does the bakery open?",
        "answer": "6 AM",
        "alternatives": ["15 years", "$3"],
        "answer_type": "string"
    },
    {
        "id": "di-016",
        "prompt": "Mountain A is taller than Mountain B. Mountain B is taller than Mountain C. Mountain A has snow year-round. Mountain B is popular for hiking. Mountain C is in a national park. Which mountain is tallest?",
        "answer": "Mountain A",
        "alternatives": ["Mountain B", "Mountain C"],
        "answer_type": "string"
    },
    {
        "id": "di-017",
        "prompt": "Amy bought 5 books. Each book cost $12. Amy's car is silver and she parked in lot B. She used a credit card. The bookstore has a cat named Whiskers. How many books did Amy buy?",
        "answer": "5",
        "alternatives": ["$12", "lot B"],
        "answer_type": "string"
    },
    {
        "id": "di-018",
        "prompt": "The doctor's appointment is at 2:00 PM. The clinic is on Main Street next to a pharmacy. The doctor's name is Dr. Chen and she specializes in cardiology. The waiting room has magazines. What time is the appointment?",
        "answer": "2:00 PM",
        "alternatives": ["Main Street", "cardiology"],
        "answer_type": "string"
    },
    {
        "id": "di-019",
        "prompt": "River A is longer than River B. River B is longer than River C. River A flows through three countries. River B has many fish. River C freezes in winter. Which river is shortest?",
        "answer": "River C",
        "alternatives": ["River A", "River B"],
        "answer_type": "string"
    },
    {
        "id": "di-020",
        "prompt": "Team A scored more points than Team B. Team A wears blue jerseys. Team B's coach has 20 years of experience. The game was on Saturday. Team B has 12 players. Which team won?",
        "answer": "Team A",
        "alternatives": ["Team B", "Saturday"],
        "answer_type": "string"
    },
    {
        "id": "di-021",
        "prompt": "The flight departs from Gate C7 at 4:15 PM. The plane holds 180 passengers. The pilot's name is Captain Reynolds and she has flown for 25 years. The airline serves peanuts. Which gate does the flight leave from?",
        "answer": "Gate C7",
        "alternatives": ["4:15 PM", "180"],
        "answer_type": "string"
    },
    {
        "id": "di-022",
        "prompt": "Recipe A takes longer to cook than Recipe B. Recipe B takes longer than Recipe C. Recipe A uses oregano. Recipe B is from Italy. Recipe C has 5 ingredients. Which recipe is quickest to make?",
        "answer": "Recipe C",
        "alternatives": ["Recipe A", "Recipe B"],
        "answer_type": "string"
    },
    {
        "id": "di-023",
        "prompt": "The museum is open Tuesday through Sunday. It was founded in 1875 and has 50,000 artifacts. The gift shop sells postcards. Admission is $15 for adults. Is the museum open on Monday?",
        "answer": "no",
        "alternatives": ["yes", "1875"],
        "answer_type": "string"
    },
    {
        "id": "di-024",
        "prompt": "House A is more expensive than House B. House B is more expensive than House C. House A has a pool. House B was built in 1990. House C has three bedrooms. Which house costs the least?",
        "answer": "House C",
        "alternatives": ["House A", "House B"],
        "answer_type": "string"
    },
    {
        "id": "di-025",
        "prompt": "The concert begins at 8 PM at the downtown arena. The band has 5 members and formed in 2005. Tickets cost $75 and the light show uses 500 lights. What time does the concert begin?",
        "answer": "8 PM",
        "alternatives": ["5", "$75"],
        "answer_type": "string"
    },
    {
        "id": "di-026",
        "prompt": "Student A scored higher than Student B on the test. Student A sits in the front row. Student B has a part-time job. The test had 50 questions. Student A's favorite subject is history. Who scored higher?",
        "answer": "Student A",
        "alternatives": ["Student B", "50"],
        "answer_type": "string"
    },
    {
        "id": "di-027",
        "prompt": "The hotel has 200 rooms. It's located near the beach and has a spa. The manager is named Patricia and she speaks three languages. Room rates start at $150. How many rooms does the hotel have?",
        "answer": "200",
        "alternatives": ["$150", "three"],
        "answer_type": "string"
    },
    {
        "id": "di-028",
        "prompt": "Lake A is deeper than Lake B. Lake B is deeper than Lake C. Lake A has trout. Lake B is popular for boating. Lake C formed from a glacier. Which lake is shallowest?",
        "answer": "Lake C",
        "alternatives": ["Lake A", "Lake B"],
        "answer_type": "string"
    },
    {
        "id": "di-029",
        "prompt": "The bus arrives every 15 minutes. The bus stop has a bench and a shelter. The route goes downtown. The driver wears a uniform. Today is Wednesday. How often does the bus arrive?",
        "answer": "every 15 minutes",
        "alternatives": ["Wednesday", "downtown"],
        "answer_type": "string"
    },
    {
        "id": "di-030",
        "prompt": "Tree A is older than Tree B. Tree B is older than Tree C. Tree A is an oak. Tree B was planted by the school. Tree C has yellow leaves in fall. Which tree is youngest?",
        "answer": "Tree C",
        "alternatives": ["Tree A", "Tree B"],
        "answer_type": "string"
    },
]

# Family D: Explanation-First Tasks
EXPLANATION_ITEMS = [
    {
        "id": "ex-001",
        "prompt": "A plant in a dark room grows tall and thin with pale leaves. Which explanation best accounts for this? A) The plant is receiving too much water. B) The plant is stretching toward any available light (etiolation). C) The plant needs more fertilizer.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-002",
        "prompt": "A city experiences increased traffic congestion despite building more highways. Which explanation best accounts for this? A) Drivers are getting lost more often. B) More road capacity induced more driving (induced demand). C) The highways were built in the wrong places.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-003",
        "prompt": "A person yawns and then others nearby also yawn. Which explanation best accounts for this? A) The air quality suddenly changed. B) Yawning is socially contagious, possibly related to empathy. C) Everyone became tired at exactly the same moment.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-004",
        "prompt": "A company raises prices by 10% and sales remain unchanged. Which explanation best accounts for this? A) Customers didn't notice the price change. B) The product has few substitutes and demand is inelastic. C) Competitors raised their prices by 20%.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-005",
        "prompt": "Birds fly south before winter arrives. Which explanation best accounts for this? A) Birds can predict the weather perfectly. B) Changing daylight hours trigger migration instincts. C) Birds follow the sun because they like warmth.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-006",
        "prompt": "Wet clothes dry faster on a windy day than a calm day. Which explanation best accounts for this? A) Wind heats up the clothes. B) Wind carries away water vapor, maintaining evaporation rate. C) Wind squeezes water out of fabric.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-007",
        "prompt": "A student performs worse on tests in the afternoon than morning. Which explanation best accounts for this? A) The afternoon tests are always harder. B) Cognitive resources and attention may deplete throughout the day. C) The classroom is darker in the afternoon.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-008",
        "prompt": "Experienced chess players can memorize board positions quickly but not random piece placements. Which explanation best accounts for this? A) They have perfect photographic memory. B) They recognize patterns and chunks from experience. C) Random placements are harder to see.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-009",
        "prompt": "A child learns to avoid a hot stove after touching it once. Which explanation best accounts for this? A) Parents told them it would be hot. B) The painful experience created a strong negative association. C) The child could see the heat visually.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-010",
        "prompt": "Food tastes bland when you have a cold. Which explanation best accounts for this? A) The food's temperature changed. B) Nasal congestion reduces smell, which affects taste perception. C) Cold viruses attack taste buds directly.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-011",
        "prompt": "A balloon shrinks when put in a refrigerator. Which explanation best accounts for this? A) The rubber becomes tighter in cold. B) Gas molecules slow down and take up less space when cooled. C) The refrigerator sucks air out of balloons.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-012",
        "prompt": "Eyewitness testimony is often unreliable. Which explanation best accounts for this? A) Most witnesses intentionally lie. B) Memory is reconstructive and susceptible to suggestion. C) Courts ask the wrong questions.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-013",
        "prompt": "Prices rise during a natural disaster. Which explanation best accounts for this? A) Store owners become more greedy during disasters. B) Supply decreases while demand increases. C) Money becomes worth less during disasters.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-014",
        "prompt": "A metal lid stuck on a jar opens after running hot water on it. Which explanation best accounts for this? A) The water lubricates the lid. B) Metal expands when heated, loosening the seal. C) Hot water dissolves the glue on the lid.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-015",
        "prompt": "People often feel happier after exercising. Which explanation best accounts for this? A) Exercise makes you forget your problems. B) Physical activity releases endorphins that improve mood. C) Gyms play happy music.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-016",
        "prompt": "A car's fuel efficiency decreases at very high speeds. Which explanation best accounts for this? A) Engines work less hard at high speeds. B) Air resistance increases dramatically with speed. C) Gasoline evaporates faster when driving fast.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-017",
        "prompt": "People are more likely to help someone in need when alone than in a crowd. Which explanation best accounts for this? A) Crowds make people aggressive. B) Diffusion of responsibility reduces individual action. C) Helpers want to be seen as heroes.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-018",
        "prompt": "Milk spoils faster when left out than when refrigerated. Which explanation best accounts for this? A) Refrigerators kill bacteria. B) Cold temperatures slow bacterial growth. C) Refrigerator light keeps milk fresh.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-019",
        "prompt": "A candle flame flickers in a draft but a light bulb doesn't. Which explanation best accounts for this? A) Light bulbs are heavier than flames. B) Flames depend on continuous combustion affected by air flow; bulbs use electricity. C) Candles are less powerful than bulbs.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-020",
        "prompt": "A person remembers the beginning and end of a list better than the middle. Which explanation best accounts for this? A) The middle items are harder words. B) Primacy and recency effects in memory. C) People skip reading the middle.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-021",
        "prompt": "Salt lowers the freezing point of water. Which explanation best accounts for this? A) Salt makes water heavier. B) Dissolved salt disrupts water molecule bonding. C) Salt absorbs the cold.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-022",
        "prompt": "A person procrastinates more on important tasks. Which explanation best accounts for this? A) Important tasks take longer. B) High stakes create anxiety that triggers avoidance. C) The person is lazy about important things only.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-023",
        "prompt": "Wounds heal faster when kept moist than dry. Which explanation best accounts for this? A) Water fills in the wound. B) Moist environments help cells migrate and regenerate. C) Dry wounds are attacked by bacteria more.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-024",
        "prompt": "A cup of coffee gets cold faster in a metal mug than ceramic. Which explanation best accounts for this? A) Metal mugs are thinner. B) Metal conducts heat away from the coffee more efficiently. C) Ceramic traps caffeine which creates heat.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-025",
        "prompt": "Children learn languages more easily than adults. Which explanation best accounts for this? A) Children have more free time. B) Young brains have greater neuroplasticity for language acquisition. C) Adults are too distracted to learn.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-026",
        "prompt": "A car battery dies more often in winter. Which explanation best accounts for this? A) Batteries prefer warm weather. B) Cold reduces chemical reaction efficiency in batteries. C) Winter has fewer hours of daylight.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-027",
        "prompt": "People often perform better with moderate anxiety than no anxiety. Which explanation best accounts for this? A) Anxiety makes people smarter. B) Moderate arousal optimizes alertness and focus (Yerkes-Dodson law). C) Anxious people study more.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-028",
        "prompt": "Ice floats on water. Which explanation best accounts for this? A) Ice is made of lighter water molecules. B) Ice is less dense than liquid water due to hydrogen bonding structure. C) Cold things always float.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-029",
        "prompt": "A broken window in a neighborhood leads to more vandalism. Which explanation best accounts for this? A) The window attracts vandals. B) Visible disorder signals that rules aren't enforced (broken windows theory). C) Vandals like the sound of breaking glass.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
    {
        "id": "ex-030",
        "prompt": "Popcorn pops when heated. Which explanation best accounts for this? A) Heat makes the kernels jump. B) Water inside the kernel turns to steam and pressure bursts the hull. C) The oils in popcorn explode when hot.",
        "answer": "B",
        "alternatives": ["A", "C"],
        "answer_type": "choice"
    },
]


def write_dataset(items: list[dict], family: str) -> None:
    """Write a dataset file."""
    outpath = ROOT / "eval" / "datasets" / f"exp005_{family}.jsonl"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} items to {outpath}")


def main() -> int:
    # Write all datasets
    write_dataset(COMMONSENSE_ITEMS, "commonsense")
    write_dataset(CAUSAL_ITEMS, "causal")
    write_dataset(DISTRACTOR_ITEMS, "distractor")
    write_dataset(EXPLANATION_ITEMS, "explanation")

    # Write README
    readme = """# Experiment 005 Datasets

## Generation Method
These datasets were generated deterministically for the semantic ambiguity stress test (Experiment 005).

## Families

### commonsense (30 items)
Everyday physical and social reasoning tasks.
- Items test basic understanding of physical causation and social conventions
- Each has one correct answer and plausible alternatives

### causal (30 items)
Multi-step story reasoning with cause-effect chains.
- Items present narratives with clear causal progressions
- Tests ability to track consequences over multiple steps

### distractor (30 items)
Word problems with irrelevant details and distractors.
- Items include extraneous information to test focus
- Correct answers require filtering relevant from irrelevant details

### explanation (30 items)
Hypothesis selection tasks (multiple choice A/B/C).
- Items present phenomena and ask for best explanation
- Tests ability to evaluate competing explanations

## Format
Each line is a JSON object with:
- id: unique identifier (format: <family>-<num>)
- prompt: the question/task
- answer: the gold answer
- alternatives: plausible but wrong answers
- answer_type: "string" or "choice"

## Seed
Datasets are static and deterministic (no random generation).
"""
    readme_path = ROOT / "eval" / "datasets" / "exp005_README.md"
    readme_path.write_text(readme)
    print(f"Wrote README to {readme_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
