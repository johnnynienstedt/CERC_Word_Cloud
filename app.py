import streamlit as st
import re
import nltk
import string
import random
import itertools
import contractions
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter
from scipy.ndimage import binary_dilation
import io
import os
import base64

def load_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Page config
st.set_page_config(page_title="Word Cloud Generator", layout="centered")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Font path - assumes font file is in same directory as script
FONT_PATH = 'Gnuolane Rg.otf'

def get_words(file_content, cloud_size, hidden_words=None):
    """Extract and process words from file content"""
    
    if hidden_words is None:
        hidden_words = set()

    APOSTROPHES = [
        '\u2019',  # ’
        '\u2018',  # ‘
        '\u02BC',  # ʼ
        '\u02BB',
        '\u2032',  # ′ prime
        '\uFF07',  # ＇ fullwidth apostrophe
    ]
    
    # separate each word and eliminate numbers and words less than 4 letters long
    all_words = []
    acronyms = set()
    for line in file_content.split('\n'):
    
        # Normalize apostrophes BEFORE contractions.fix()
        for a in APOSTROPHES:
            line = line.replace(a, "'")
        
        fixed_line = contractions.fix(line)
        
        temp = fixed_line.split()
    
        for idx, x in enumerate(temp):

            x = re.sub(r'^[\W_]+|[\W_]+$', '', x)
            x = re.sub('[0-9]', '', x)
            
            # Strip punctuation at the end
            while x.endswith(('.', ',', '!', '?', "'")):
                x = x[:-1]
        
            # Handle possessive endings
            if x.endswith("'s"):
                x = x[:-2]
        
            if '+' in x:
                x = x.replace('+', ' ')
        
            if x.startswith('\ufeff'):
                x = x.replace('\ufeff', '')

            # Detect acronyms: all letters uppercase, length <= 5
            if x.isupper() and len(x) <= 5:
                acronyms.add(x)
            
            # Force first word of each line to be lowercase
            if idx == 0 and len(x) > 0:
                x = x[0].lower() + x[1:]
                
            if len(x) > 1:
                all_words.append(x)
    
    # eliminate spaces and punctuation
    for item in ['', ',', '-', '–', "'", ''', ''', '"', '"', '"',]:
        if item in all_words:
            all_words.remove(item)

    # list of common words
    common_words = ('the', 'a', 'at', 'there',	'some',	'my', 'of', 'be', 'use',	
                    'her', 'than', 'and', 'this', 'an', 'would', 'first', 'have',
                    'each', 'make', 'to', 'from', 'which', 'like', 'been', 'in',
                    'or', 'she', 'him', 'is', 'one', 'do', 'into'	, 'who', 'you',
                    'had',	'how', 'time', 'that', 'by', 'their', 'has'	, 'its',
                    'it', 'word', 'if', 'look', 'now', 'he', 'but', 'will', 'two',
                    'find', 'was', 'not', 'up', 'more', 'long', 'for', 'what', 
                    'other', 'down', 'on', 'all', 'about', 'go', 'day', 'are',
                    'were', 'out', 'see', 'did', 'with', 'when', 'then', 'no', 
                    'come', 'his', 'your', 'them', 'way', 'made', 'they', 'can',
                    'these', 'could', 'may', 'i', 'said', 'so', 'part', 'across',
                    'again', 'began', 'begin', 'going', 'should', 'while', 'those',
                    'still', 'bring', 'after', 'before', 'around', 'able', 'above',
                    'although', 'already', 'away', 'behind', 'below', 'came',
                    'else', 'instead', 'however', 'just', 'iteslf', 'himself',
                    'herself', 'know', 'many', 'over', 'perhaps', 'only', 'much',
                    'sure', 'several', 'take', 'took', 'toward', 'especially',
                    'eventually', 'never', 'need', 'myself', 'most', 'very',
                    'also', 'actually', 'against', 'almost', 'last', 'back', 'goes',
                    'always', 'felt', 'feel', 'since', 'probably', 'knew', 'even',
                    'think', 'until', 'wait', 'maybe', 'want', 'cannot', 'does',
                    'another', 'being', 'once', 'through', 'though', 'tell', 'city',
                    'done', 'must', 'here', 'every', 'thing', 'such', 'things', 
                    'really', 'because', 'where', 'without', 'themselves')
    
    # eliminate common words
    filtered_words = []
    for word in all_words:
        if word.lower() not in common_words:
            filtered_words.append(word)
    all_words = filtered_words

    # STEP 1: Detect bigrams
    bigram_counts = Counter()
    for i in range(len(all_words) - 1):
        bigram = (all_words[i], all_words[i+1])
        bigram_counts[bigram] += 1
    
    # Count individual words
    word_counts = Counter(all_words)
    
    # Find bigrams that appear more often than the sum of individuals
    detected_bigrams = []
    words_in_bigrams = set()  # Track which words are part of bigrams
    
    for (w1, w2), bigram_freq in bigram_counts.items():
        individual_freq = word_counts[w1] + word_counts[w2]
        
        # If bigram appears more than sum of individuals, it's a strong collocation
        if bigram_freq > individual_freq - 2*bigram_freq and bigram_freq > 5:
            detected_bigrams.append((w1, w2))
            words_in_bigrams.add(w1)
            words_in_bigrams.add(w2)
    
    # Replace sequences in all_words with bigram tokens
    processed_words = []
    i = 0
    while i < len(all_words):
        if i < len(all_words) - 1:
            potential_bigram = (all_words[i], all_words[i+1])
            if potential_bigram in detected_bigrams:
                processed_words.append(f"{all_words[i]} {all_words[i+1]}")
                i += 2
                continue
        processed_words.append(all_words[i])
        i += 1
    
    # Now remove standalone instances of words that are part of bigrams
    all_words = [word for word in processed_words if ' ' in word or word not in words_in_bigrams]
    all_words = [word for word in all_words if len(word) >= 4]
    
    # STEP 2: Identify proper nouns (words that appear capitalized more than 70% of the time)
    word_case_counts = {}
    for i, word in enumerate(all_words):
        # Skip bigrams
        if ' ' in word:
            continue
        lower_word = word.lower()
        if lower_word not in word_case_counts:
            word_case_counts[lower_word] = {'caps': 0, 'total': 0}

        # Check if word is capitalized
        if word and word[0].isupper():
            word_case_counts[lower_word]['caps'] += 1
        word_case_counts[lower_word]['total'] += 1
    
    # Determine which words are proper nouns
    proper_nouns = set()
    for word, counts in word_case_counts.items():
        if counts['total'] >= 3 and counts['caps'] / counts['total'] > 0.7:
            proper_nouns.add(word)
    
            
    # STEP 3: Standardize capitalization (but preserve proper nouns)
    normalized_words = []
    for word in all_words:
        if ' ' in word:  # Bigram
            normalized_words.append(word)
        elif word.lower() in proper_nouns:
            # Keep proper noun capitalized
            normalized_words.append(word.title())
        elif word.upper() in acronyms:
            normalized_words.append(word)
        else:
            normalized_words.append(word.lower())
    
    all_words = normalized_words
    
    # count occurences of each word
    words = Counter(all_words)
    
    # STEP 4: Standardize pluralization (skip bigrams and proper nouns)
    lemmatizer = WordNetLemmatizer()

    # Group words by their lemma
    lemma_groups = {}
    for word in words:
        if ' ' in word:
            # Don't lemmatize bigrams
            lemma = word
        else:
            lemma = lemmatizer.lemmatize(word)
        if lemma not in lemma_groups:
            lemma_groups[lemma] = []
        lemma_groups[lemma].append(word)
    
    # For each group, find the most frequent form
    all_words_normalized = []
    for word in all_words:
        if ' ' in word or word.lower() in proper_nouns:
            lemma = word
        else:
            lemma = lemmatizer.lemmatize(word)
        # Get the most frequent word in this lemma group
        most_frequent = max(lemma_groups[lemma], key=lambda w: words[w])
        all_words_normalized.append(most_frequent)
    
    # count occurences of each word
    words = Counter(all_words_normalized)

    # STEP 5: Stem-based grouping (skip bigrams and proper nouns)
    stemmer = PorterStemmer()
    stem_groups = {}
    
    # Group words by their stem
    for word, freq in words.items():
        if ' ' in word or word.lower() in proper_nouns:
            stem = word  # Don't stem bigrams or proper nouns
        else:
            stem = stemmer.stem(word)
        if stem not in stem_groups:
            stem_groups[stem] = []
        stem_groups[stem].append((word, freq))
    
    # For each group, select the most frequent word
    result = {}
    for stem, group in stem_groups.items():
        most_frequent_word = max(group, key=lambda x: x[1])[0]
        total_freq = sum(f for _, f in group)
        result[most_frequent_word] = total_freq
    
    # Sort by frequency
    result = dict(sorted(result.items(), key = lambda x:x[1], reverse = True))
    
    # Filter out hidden words (case-insensitive) and then truncate to correct size
    # This ensures we always get cloud_size words (if available) after filtering
    if hidden_words:
        result = {word: freq for word, freq in result.items() 
                  if word.lower() not in hidden_words}
    
    # Truncate to cloud_size AFTER filtering
    result = dict(itertools.islice(result.items(), cloud_size))
        
    return result

def generate_word_cloud(word_counts, width=1200, height=800, font_path=None, 
                        min_font_size=15, max_font_size=150, margin=4):
    """
    Generate a word cloud with frequency-based sizing and pixel-perfect placement.
    """
    
    def string_to_lex_value(s):
        """Convert a string to a numerical value that preserves alphabetical ordering."""
        s = ''.join(c for c in s.lower() if c.isalpha())
        if not s:
            return 0.0
        
        value = 0.0
        for i, char in enumerate(s):
            letter_value = ord(char) - ord('a') + 1  # a=1, b=2, ..., z=26
            value += letter_value / (26 ** (i + 1))
        
        return value
    
    # Create white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Create occupancy mask (tracks which pixels are occupied)
    occupancy = np.zeros((height, width), dtype=bool)
    
    # Prepare word list with scoring
    words_data = []
    max_count = max(word_counts.values())
    min_count = min(word_counts.values())
    
    for word, count in word_counts.items():
        # Normalize count (0-1)
        norm_count = (count - min_count) / (max_count - min_count) if max_count > min_count else 1
        
        # Alphabetical score (0-1, where 0 is 'a' and 1 is 'z')
        alpha_score = string_to_lex_value(word)
                
        # Font size calculation
        font_size = int(min_font_size + (max_font_size - min_font_size) * (norm_count ** 1.2))
        
        words_data.append({
            'word': word,
            'count': count,
            'font_size': font_size,
            'alpha_score': alpha_score
        })
    
    # Sort by count (high to low)
    words_data.sort(key=lambda x: (-x['count']))
    
    # Load font
    def get_font(size):
        return ImageFont.truetype(font_path, size)
    
    def get_word_mask(word, font_size, margin):
        """Return both original and dilated masks for a word."""
        font = get_font(font_size)
    
        # Get bbox to determine size needed
        bbox = ImageDraw.Draw(Image.new('L', (1, 1), 0)).textbbox((0, 0), word, font=font)
        word_width = bbox[2] - bbox[0]
        word_height = bbox[3] - bbox[1]
    
        if word_width <= 0 or word_height <= 0:
            return None, None, 0, 0, (0, 0)
    
        # Create temp image and draw text at (0, 0) to get the actual drawn pixels
        temp_img = Image.new('L', (word_width + abs(bbox[0]) + 10, word_height + abs(bbox[1]) + 10), 255)
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text((0, 0), word, fill=0, font=font)
        
        # Get the actual bbox of what was drawn
        temp_array = np.array(temp_img)
        drawn_pixels = temp_array < 128
        rows = np.any(drawn_pixels, axis=1)
        cols = np.any(drawn_pixels, axis=0)
        
        if not rows.any() or not cols.any():
            return None, None, 0, 0, (0, 0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Crop to actual content
        original_mask = drawn_pixels[y_min:y_max+1, x_min:x_max+1]
        
        # Store the offset for alignment
        text_offset = (x_min, y_min)
        
        # Create dilated mask for margin/occupancy
        if margin > 0:
            # PAD the mask first so dilation can expand beyond original boundaries
            padded_mask = np.pad(original_mask, margin, mode='constant', constant_values=False)
            
            # Now dilate the padded mask
            dilated_mask = binary_dilation(padded_mask, iterations=margin)
            
            # Update text_offset to account for padding
            adjusted_text_offset = (text_offset[0] + margin, text_offset[1] + margin)
            
        else:
            dilated_mask = original_mask
            adjusted_text_offset = text_offset
    
        return original_mask, dilated_mask, dilated_mask.shape[1], dilated_mask.shape[0], adjusted_text_offset
    
    def check_collision(x, y, dilated_mask):
        """Check if placing dilated_mask at (x,y) would collide with existing words"""
        mask_h, mask_w = dilated_mask.shape
        
        # Check bounds
        if x < 0 or y < 0 or x + mask_w > width or y + mask_h > height:
            return True
        
        # Get the exact region where this word would be placed
        region = occupancy[y:y+mask_h, x:x+mask_w]
        
        # Verify shapes match (they should)
        if region.shape != dilated_mask.shape:
            return True
        
        # Pixel-perfect collision: check where BOTH dilated_mask AND region are True
        collision = np.any(dilated_mask & region)
        
        return collision
    
    def place_word_mask(x, y, dilated_mask):
        """Mark the dilated mask area as occupied"""
        mask_h, mask_w = dilated_mask.shape
        occupancy[y:y+mask_h, x:x+mask_w] |= dilated_mask
    
    def find_position(word, font_size, alpha):
        """Find a position for the word using pixel-perfect collision detection"""
        result = get_word_mask(word, font_size, margin)
        
        if result[0] is None:
            return None
        
        original_mask, dilated_mask, dilated_w, dilated_h, text_offset = result
        
        # Start using alpha order
        buffer = 0.2 * width
        center_x = (width - buffer) * alpha + buffer*2/3
        center_y = height // 2
        
        # Spiral search pattern
        y_offset = 0
        x_offset = 0
        
        valid = {}
        for attempt_x in range(width):
            for attempt_y in range(height):
            
                # Calculate desired center position for the dilated mask
                desired_mask_center_x = int(center_x + x_offset)
                desired_mask_center_y = int(center_y + y_offset)
                
                # Calculate dilated mask position (top-left corner)
                mask_x = desired_mask_center_x - dilated_w // 2
                mask_y = desired_mask_center_y - dilated_h // 2
                
                # Text position is mask position minus the offset
                text_x = mask_x - text_offset[0]
                text_y = mask_y - text_offset[1]
                
                # Check collision using dilated mask
                if not check_collision(mask_x, mask_y, dilated_mask):
                    alpha_distance = np.sqrt((text_x - center_x)**2 + (text_y - center_y)**2)
                    packing_distance = np.sqrt((text_x - width/2)**2 + (text_y - height/2)**2)
                    valid[(text_x, text_y, mask_x, mask_y)] = (alpha_distance + 2*packing_distance, dilated_mask)
                
                y_offset = (-1)**attempt_y * (attempt_y+random.random()*height/200)
                x_offset = random.random() * width/10
                
                if len(valid) >= 5:
                    break
                        
        if len(valid) > 0:
            best_pos = min(valid.keys(), key=lambda k: valid[k][0])
            score, mask = valid[best_pos]
            return (*best_pos, mask)
                        
        return None
    
    # Place words
    placed_count = 0
    for i, word_data in enumerate(words_data):
        word = word_data['word']
        alpha = word_data['alpha_score']
        font_size = word_data['font_size']
        
        result = find_position(word, font_size, alpha)
        if result:
            text_x, text_y, mask_x, mask_y, dilated_mask = result
            font = get_font(font_size)
            # Draw the actual text
            draw.text((text_x, text_y), word, fill='black', font=font)
            # Mark the dilated area as occupied
            place_word_mask(mask_x, mask_y, dilated_mask)
            placed_count += 1
    
    return img, placed_count, len(words_data)

def crop_to_content(img, margin=5):
    """Crop image to content with specified margin"""
    # Convert to numpy array
    img_array = np.array(img)
    
    # Find non-white pixels (assuming white background is 255,255,255)
    non_white = np.any(img_array != 255, axis=2)
    
    # Find bounding box
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    
    if not rows.any() or not cols.any():
        return img  # Return original if no content
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add margin
    y_min = max(0, y_min - margin)
    y_max = min(img_array.shape[0], y_max + margin + 1)
    x_min = max(0, x_min - margin)
    x_max = min(img_array.shape[1], x_max + margin + 1)
    
    # Crop
    cropped = img.crop((x_min, y_min, x_max, y_max))
    return cropped

# Streamlit UI
logo_base64 = load_image_as_base64("Big-C-Red.png")

st.markdown(
    f"""
    <h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
        <img src="data:image/png;base64,{logo_base64}" 
             style="height: 80px; vertical-align: middle;">
        Word Cloud Generator
    </h1>
    """,
    unsafe_allow_html=True
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Make file uploader area taller */
    [data-testid="stFileUploader"] > section > button {
        padding: 3rem 2rem !important;
        min-height: 150px;
    }
    /* Make dropdown narrower */
    div[data-baseweb="select"] {
        max-width: 150px;
    }
    </style>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Drag and drop file here", type=['txt'], label_visibility="visible")

# Cloud size selector and word exclusion input side by side
col1, col2 = st.columns([1, 3])

with col1:
    cloud_size = st.selectbox(
        "Number of words in cloud",
        options=[30, 40, 50, 60, 70, 80, 90, 100],
        index=1  # Default to 40
    )

with col2:
    # Initialize hidden words in session state
    if 'hidden_words' not in st.session_state:
        st.session_state.hidden_words = set()
    
    col2a, col2b = st.columns([3, 1])
    with col2a:
        word_to_hide = st.text_input(
            "Exclude word or phrase",
            key="word_to_hide",
            placeholder="Enter word to hide..."
        )
    with col2b:
        st.markdown("<div style='margin-top: 1.85rem;'></div>", unsafe_allow_html=True)
        if st.button("Hide", use_container_width=True):
            if word_to_hide and word_to_hide.strip():
                st.session_state.hidden_words.add(word_to_hide.strip().lower())
                st.session_state.hide_word_input = ""
                st.rerun()
    
    # Display hidden words if any exist
    if st.session_state.hidden_words:
        hidden_list = ", ".join(sorted(st.session_state.hidden_words))
        st.caption(f"Hidden: {hidden_list}")
        if st.button("Clear all hidden words", key="clear_hidden"):
            st.session_state.hidden_words = set()
            st.rerun()

# Check if font exists
if not os.path.exists(FONT_PATH):
    st.error(f"⚠️ Font file '{FONT_PATH}' not found. Please ensure it's in the same directory as this script.")
    st.stop()

# Generate and Download buttons side by side
col1, col2 = st.columns([1, 1])

# Generate button
if uploaded_file is not None:
    with col1:
        generate_clicked = st.button("Generate Cloud", type="primary", use_container_width=True)
    
    if generate_clicked:
        # Read file
        file_content = uploaded_file.read().decode('cp1252')
        
        # Check if file has enough content
        if len(file_content.strip()) < 100:
            st.warning("⚠️ The uploaded file appears to be too short. Please upload a longer text file.")
        else:
            with st.spinner('Generating word cloud...'):
                try:
                    # Process words with hidden words filter
                    word_counts = get_words(file_content, cloud_size, st.session_state.hidden_words)
                    
                    if len(word_counts) == 0:
                        st.error("❌ No valid words found in the file. Please check your text file.")
                    else:
                        # Generate cloud
                        img, placed_count, total_words = generate_word_cloud(
                            word_counts,
                            width=1200,
                            height=800,
                            font_path=FONT_PATH,
                            min_font_size=15,
                            max_font_size=150,
                            margin=3
                        )
                        
                        # Crop to content
                        img_cropped = crop_to_content(img, margin=5)
                        
                        # Store in session state
                        st.session_state.word_cloud = img_cropped
                        st.session_state.placed_count = placed_count
                        st.session_state.total_words = total_words
                        
                        st.success(f"✅ Successfully placed {placed_count} out of {total_words} words!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error generating word cloud: {str(e)}")

# Display word cloud if it exists
if 'word_cloud' in st.session_state:
    st.image(st.session_state.word_cloud, use_container_width=True)
    
    # Download button (only show when cloud exists)
    if uploaded_file is not None:
        buf = io.BytesIO()
        st.session_state.word_cloud.save(buf, format='PNG')
        buf.seek(0)
        
        with col2:
            st.download_button(
                label="Download Cloud",
                data=buf,
                file_name="word_cloud.png",
                mime="image/png",
                type="primary",
                use_container_width=True
            )

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. **Upload a text file** (.txt format) using the file uploader above
    2. **Select the number of words** you want in your cloud (30-100)
    3. **Click the generate button** to create your word cloud
    4. **Re-shuffle** by clicking the button again without changing parameters
    5. **Download** your word cloud using the download button
    
    **Note:** The word cloud uses intelligent text processing including:
    - Bigram detection for common phrases
    - Proper noun identification and capitalization
    - Lemmatization and stemming for word grouping
    - Common word filtering
    """)
