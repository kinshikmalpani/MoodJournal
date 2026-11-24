"""AI-authored diagnostic fixtures, NOT human-labelled journal evaluation."""
import json
from pathlib import Path

CASES = [
    ('direct', ['anger'], 'I am furious that my colleague took credit for my work.'),
    ('direct', ['anger'], 'Being interrupted again left me irritated and resentful.'),
    ('negation', ['anger'], 'I am not scared of him. I am angry that he lied to me.'),
    ('journal', ['anger'], 'I spent the afternoon fixing a mistake that was not mine. When the manager blamed me in front of everyone, I wanted to shout. Even after getting home I was still furious about how unfair it was.'),
    ('direct', ['fear'], 'I am terrified about what the test results might show.'),
    ('direct', ['fear'], 'Every unfamiliar noise tonight makes me nervous.'),
    ('negation', ['fear'], 'I am not excited about tomorrow. I am frightened that everything will go wrong.'),
    ('journal', ['fear'], 'The presentation is tomorrow morning. I rehearsed twice and packed my bag, but my hands are still shaking. I keep imagining that I will forget everything and embarrass myself in front of the room.'),
    ('direct', ['joy'], 'I feel delighted and proud after finishing my first project.'),
    ('direct', ['joy'], 'We won the match and I cannot stop smiling.'),
    ('negation', ['joy'], 'I am not miserable anymore. Today I feel cheerful and hopeful.'),
    ('journal', ['joy'], 'I finally submitted the assignment that had taken weeks. My friends came over and we cooked dinner together. I laughed so much tonight, and I went to bed feeling genuinely happy about the day.'),
    ('direct', ['love'], 'I feel deeply affectionate toward my partner.'),
    ('direct', ['love'], 'Holding my daughter close filled me with tenderness.'),
    ('negation', ['love'], 'I do not resent my brother. I adore him and treasure our closeness.'),
    ('journal', ['love'], 'Grandma called to ask whether I had eaten. We talked for an hour about small things. Thinking about how much she cares made me feel warm and connected, and I wanted to tell her how dearly I love her.'),
    ('direct', ['sadness'], 'I feel lonely and heartbroken after saying goodbye.'),
    ('direct', ['sadness'], 'Everything feels empty since my friend moved away.'),
    ('negation', ['sadness'], 'I am not happy about leaving. I feel miserable and keep crying.'),
    ('journal', ['sadness'], 'I saw an old photograph of our group at lunch. Everyone has moved to different cities now. The apartment felt very quiet tonight, and I missed them so much that I cried before going to sleep.'),
    ('direct', ['surprise'], 'I was astonished to discover a hidden room behind the bookcase.'),
    ('direct', ['surprise'], 'That completely unexpected announcement left me speechless.'),
    ('negation', ['surprise'], 'I was not expecting a visit at all. Seeing her at the door stunned me.'),
    ('journal', ['surprise'], 'The morning followed its usual routine until an old classmate walked into the office. I had assumed he lived overseas. For a moment I could only stare because the coincidence was so unexpected.'),
    ('negation', ['sadness'], 'People think I am enjoying the party, but I am not. I feel painfully lonely.'),
    ('negation', ['joy'], 'I thought I would be frightened, but I was not. I loved the ride and felt exhilarated.'),
    ('negation', ['fear'], 'There is nothing comforting about the message. I am anxious about what happens next.'),
    ('negation', ['anger'], 'I do not find the joke funny. I feel offended and furious.'),
    ('neutral', [], 'I took the bus at eight, attended two classes, and bought rice on the way home.'),
    ('neutral', [], 'The meeting is scheduled for Tuesday at three in room twelve.'),
    ('neutral', [], 'I put the receipts in a folder and charged my phone.'),
    ('neutral', [], 'Today I washed the dishes. Then I folded the laundry and checked the calendar.'),
    ('neutral', [], 'The notebook has eighty pages and a blue cover.'),
    ('neutral', [], 'Lunch was at one. The next lecture began at two and finished at three.'),
    ('mixed', ['fear', 'joy'], 'I am frightened about the interview tomorrow. Tonight I feel delighted that my friends came to celebrate my birthday.'),
    ('mixed', ['anger', 'love'], 'I am furious about the unfair decision at work. Holding my child tonight filled me with tenderness and love.'),
    ('mixed', ['sadness', 'joy'], 'I cried because I miss my old friends. Later I felt thrilled when I learned that I passed the exam.'),
    ('mixed', ['surprise', 'fear'], 'The unexpected invitation left me astonished. Now I feel terrified about travelling alone.'),
    ('mixed', ['love', 'sadness'], 'I feel deeply loved when my partner holds my hand. I also feel heartbroken about our upcoming separation.'),
    ('mixed', ['anger', 'joy'], 'The rude comment made me furious this morning. Winning our match this evening made me incredibly happy.'),
]


def main():
    rows = [{'id': f'dev-{i:03}', 'category': category, 'expected_emotions': labels, 'text': text, 'source': 'AI-authored development fixture; labels require independent human review'} for i, (category, labels, text) in enumerate(CASES, 1)]
    # Explicit long-input retention check; not intended to imitate natural journals.
    rows.append({'id': 'dev-long', 'category': 'long', 'expected_emotions': ['sadness', 'joy'], 'text': 'I feel heartbroken and lonely. ' + 'I checked the timetable and wrote down the next appointment. ' * 35 + 'Tonight I feel delighted and excited about my achievement.', 'source': 'AI-authored artificial length stress test'})
    path = Path(__file__).with_name('journal_development.jsonl')
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows)} development cases to {path}')


if __name__ == '__main__':
    main()
