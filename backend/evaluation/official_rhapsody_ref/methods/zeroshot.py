from methods.base import BaseMethod
from openai import OpenAI
from google import genai
from google.genai import types
import re

# system and user prompts for text-only setting
SYSTEM_PROMPT = """You are an assistant trained to analyze summaries of podcast segments and predict listener behavior. Your task is to identify segments that are most likely to be replayed, rank them by likelihood, and provide concise and specific rationales. Follow these guidelines when answering:

1. Rank segments in descending order of replay likelihood.
2. Include a segment only if there is a highly obvious reason it might be replayed, excluding uncertain segments.
3. If no segments meet the criteria, leave the 'Answer' item empty.
4. Respond in the following format:
- Segment X: (one-line rationale for why it is replayed)
- Segment Y: (one-line rationale for why it is replayed)
...
- Segment Z: (one-line rationale for why it is replayed)
- Answer: X, Y, ..., Z"""

USER_PROMPT = """Analyze the following podcast segments:
- Title: {title}
{segments}
Which segments are most likely to be replayed?"""

# system prompt for text+audio setting
SYSTEM_PROMPT_AUDIO = """You are an assistant trained to analyze summaries of podcast segments and predict listener behavior. You will be provided the summaries of podcast segments as well as the original podcast audio. Your task is to identify segments that are most likely to be replayed, rank them by likelihood, and provide concise and specific rationales. You should use both the provided summary text and the audio to make your decision, paying attention to the content of the text and audio as well as prosodic and emotional cues from the audio. Follow these guidelines when answering:

1. Rank segments in descending order of replay likelihood.
2. Include a segment only if there is a highly obvious reason it might be replayed, excluding uncertain segments.
3. If no segments meet the criteria, leave the 'Answer' item empty.
4. Respond in the following format:
- Segment X: (one-line rationale for why it is replayed)
- Segment Y: (one-line rationale for why it is replayed)
...
- Segment Z: (one-line rationale for why it is replayed)
- Answer: X, Y, ..., Z"""

class ZeroShotMethod(BaseMethod):
    """Zero-shot highlight detection using a proprietary language model."""
    
    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument(
            "--model",
            type=str,
            default="gpt-4o-2024-08-06", # for Gemini, we used "gemini-2.0-flash-001" in the paper
            help="model to use for zero-shot prompting",
        )
        parser.add_argument(
            "--max_tokens",
            type=int,
            default=1024,
            help="maximum number of tokens to generate",
        )
        parser.add_argument(
            "--use_audio",
            action="store_true",
            help="additionally provide the original podcast audio to the model",
        )
    
    def __init__(self, args, train_dataset):
        self.model = args.model
        self.max_tokens = args.max_tokens

        self.use_audio = args.use_audio
        if self.use_audio and 'gemini' not in self.model:
            raise ValueError("Audio support is only available for Gemini models.")
        
        if 'gpt' in self.model:
            self.client = OpenAI()
        elif 'gemini' in self.model:
            self.client = genai.Client()
        else:
            raise ValueError(f"Unsupported model: {self.model}. We support only OpenAI and Gemini models.")

    @staticmethod
    def _parse_response(response: str) -> list[int]:
        """
        Parse segment predictions from LLM response.
        Expected format: "Answer: X, Y, Z"
        """
        # Try to find explicit "Answer:" line
        answer_matches = re.findall(r'Answer:(.*)', response)
        
        if len(answer_matches) == 1:
            # Parse explicit answer format: "Answer: X, Y, Z"
            answer_text = answer_matches[0].replace('and', ',')
            items = answer_text.split(',')
            
            predictions = []
            filter_words = ['Segments', 'Segment', '.']
            
            for item in items:
                # remove filter words and whitespace
                for word in filter_words:
                    if word in item:
                        item = item.replace(word, '')
                item = item.strip()
                
                # Skip empty or N/A items
                if item == "" or item == "N/A":
                    continue
                
                # Parse: handle single numbers, ranges, and lists
                if item.isdigit():
                    predictions.append(int(item))
                elif '-' in item:
                    start, end = map(int, item.split('-'))
                    predictions.extend(range(start, end + 1))
                elif '–' in item:
                    start, end = map(int, item.split('–'))
                    predictions.extend(range(start, end + 1))
                elif ' to ' in item:
                    start, end = map(int, item.split(' to '))
                    predictions.extend(range(start, end + 1))
                elif '/' in item:
                    indices = map(int, item.split('/'))
                    predictions.extend(indices)
                elif '&' in item:
                    indices = map(int, item.split('&'))
                    predictions.extend(indices)
            
            return predictions
        else:
            if 'no segments' in response or 'any segments' in response:
                return []
            
            # parse from rationale
            rationale_matches = re.findall(r'Segment (\d+): (.*)', response)
            if len(rationale_matches) == 0:
                return []
            
            predictions = []
            for match in rationale_matches:
                index = int(match[0])
                predictions.append(index)
            return predictions

    def predict(self, row):
        # format messages
        formatted_segments = '\n'.join([f'- Segment {i}: {item}' for i, item in enumerate(row['segment_summaries'])])
        user_prompt = USER_PROMPT.format(title=row['title'], segments=formatted_segments)
        
        if 'gpt' in self.model:
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            response = response.to_dict()
            response_text = response['choices'][0]['message']['content']
        elif 'gemini' in self.model:
            contents = [user_prompt]
            # optionally prepend audio to the contents
            if self.use_audio:
                youtube_url = f"https://www.youtube.com/watch?v={row['vid']}"
                contents = [types.Part.from_uri(file_uri=youtube_url, mime_type="audio/wav")] + contents
                
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_AUDIO if self.use_audio else SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                ),
            )
            response_text = response.text

        return self._parse_response(response_text)
