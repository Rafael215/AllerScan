import streamlit as st
import cv2
import easyocr
import re
import numpy as np
from predict_allergens import predict_allergens

st.set_page_config(page_icon="🍔", page_title="AllerScan", layout="centered")

# DARK MODE OPTION
st.sidebar.write("### Appearance")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_toggle")

# Sizes, colors, and hover animations for HomePage icons & buttons 
shared_css = """
    .stButton > button { height: 20px; border-radius: 20px; }

    .icon-box {
        width: 60px; height: 60px; border-radius: 15px; display: flex;
        justify-content: center; align-items: center; font-size: 30px;
        margin-bottom: 20px; transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .icon-box:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0px 10px 20px rgba(0,0,0,0.15);
    }
    .bg-blue { background-color: #3b82f6; color: white; }
    .bg-purple { background-color: #a855f7; color: white; }
    .bg-orange { background-color: #e67451; color: white; }
    .bg-green { background-color: #10b981; color: white; }
"""

# DYNAMIC APPEARANCE (Dark/Light Modes) 
if dark_mode:
    theme_css = """
    /* Main App Backgrounds */
    [data-testid="stAppViewContainer"] { background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%) !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important; }
    
    /* Main Text Colors */
    h1, h2, h3, h4, h5, h6, p, label, span { color: #f8fafc !important; }
    [data-testid="stSidebarNav"] ul li a span { font-size: 1.25rem !important; font-weight: 600 !important; }
    
    /* Title Boxes */
    .title-box { background-color: #334155 !important; color: #f8fafc !important; padding: 10px 40px !important; border-radius: 15px !important; display: inline-block !important; border: 1px solid #475569 !important; }
    
    /* White Borders in Dark Mode */
    [data-testid="stAppViewContainer"] div {
        border-color: #ffffff !important; 
    }

    /* Keeping the text dark while typing */
    [data-testid="stTextInput"] div[data-baseweb="input"],
    [data-testid="stTextInput"] input {
        background-color: #1e293b !important; 
        color: #ffffff !important;            
        -webkit-text-fill-color: #ffffff !important;
        border-color: #475569 !important; 
    }
    
    /* Keeping buttons the same */
    [data-testid="stButton"] button {
        background-color: #334155 !important; 
        color: #ffffff !important;            
        border: 1px solid #475569 !important; 
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1e293b !important;
        border: 2px dashed #475569 !important; 
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploadDropzone"] *,
    [data-testid="stFileUploaderDropzone"] * {
        color: #f8fafc !important;
    }
    /* Keeping the File Uploader Icon */
    [data-testid="stFileUploader"] button {
        background-color: #334155 !important; /* Dark background */
        color: #ffffff !important;            /* White text */
        border: 1px solid #475569 !important; /* border around boxes */
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background-color: #475569 !important; 
        border-color: #cbd5e1 !important;
    }



    """
else:
    theme_css = """
    /* Light Mode Background */
    [data-testid="stAppViewContainer"] { background: linear-gradient(90deg, #f1f5f9 0%, #ffffff 100%) !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #e2e8f0 0%, #cbd5e1 100%) !important; }
    
    /* Text Colors for Light Mode */
    h1, h2, h3, h4, h5, h6, p, label, span { color: #1e293b !important; }
    [data-testid="stSidebarNav"] ul li a span { font-size: 1.25rem !important; font-weight: 600 !important; }
    
    /* Title Boxes */
    .title-box { background-color: #e2e8f0 !important; color: #1e293b !important; padding: 10px 40px !important; border-radius: 15px !important; display: inline-block !important; border: 1px solid #cbd5e1 !important; }

    /* Borders for boxes in light mode */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #cbd5e1 !important; 
        border-style: solid !important;
        border-width: 1px !important;
        background-color: #ffffff !important;
        border-radius: 12px !important;
    }
    """

# Navigation Menu to direct user to profile and scanner page 
st.markdown(f"<style>\n{shared_css}\n{theme_css}\n</style>", unsafe_allow_html=True)

def homepage():
    st.markdown("""
        <div style="text-align: center;">
                <h1 class="title-box">Welcome to AllerScan</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Scan your menu to identify any potential allergens!</p>", unsafe_allow_html=True)
    
    Left_gap, col1, gap1, col2, gap2, col3, Right_gap = st.columns([1,50,10,50,10,50, 1])

    with col1:
        st.write("##")
        with st.container(border=True): 
            st.write("##### **1. Setup Profile**")
            
            st.markdown("""
            <a href="profile" target="_self" style="text-decoration: none;">
                <div class="icon-box bg-orange">
                    👤
                </div>
            </a>
        """, unsafe_allow_html=True)

    with col2:
        st.write("##") 
        with st.container(border=True):
            st.write("##### **2. Scan Menu**")
            st.markdown("""
            <a href="scan" target="_self" style="text-decoration: none;">
                <div class="icon-box bg-green">
                    ⛶
                </div>
            </a>
        """, unsafe_allow_html=True)

    with col3: 
        st.write("##")
        with st.container(border = True):
            st.write("##### **3. Get Results**")
            st.markdown("""
            <a href="scan" target="_self" style="text-decoration: none;">
                <div class="icon-box bg-blue">
                    📝
                </div>
            </a>
        """, unsafe_allow_html=True)
            
# PROFILE PAGE SETUP 
def profile():
    st.markdown("""
        <div style="text-align: center;">
                <h1 class="title-box">Profile Setup</h1>
        </div>
    """, unsafe_allow_html=True)
    st.write("---") 
    st.info("##### **Choose which Food Allergens you want AllerScan to keep track of !**")
    
    # Create the starting list of allergens if it doesn't exist
    if "allergen_list" not in st.session_state:
        st.session_state.allergen_list = ['Milk','Wheat', 'Tree Nut', 'Soy', 'Fish', 'Shellfish', 'Gluten','Egg']
        
    if "active_allergens" not in st.session_state:
        st.session_state.active_allergens = []

    st.write("### Select Allergens")
    # Creates an on/off switch and a delete button for the items in the list
    for allergen in st.session_state.allergen_list: 
        with st.container(border=True):

            keep_col, del_col = st.columns([5,1])

            with keep_col: 
                # Updates the website whenever a switch is clicked
                is_active = allergen in st.session_state.active_allergens
                
                # Put the allergen in the active list if it is turned on and takes it out if it is turned off
                toggled = st.toggle(f"{allergen}", value=is_active, key=f"toggle_{allergen}")
                
                # If user flips the switch, it adds or removes it from the active list to cross-check for allergens 
                if toggled and allergen not in st.session_state.active_allergens:
                    st.session_state.active_allergens.append(allergen)
                elif not toggled and allergen in st.session_state.active_allergens:
                    st.session_state.active_allergens.remove(allergen)

            with del_col: 
                if st.button("❌", key=f"del_{allergen}"):
                    st.session_state.allergen_list.remove(allergen)
                    # Deletes the allergen completely and reloads the page
                    if allergen in st.session_state.active_allergens:
                        st.session_state.active_allergens.remove(allergen)
                    st.rerun()

    st.write("---")
    st.write("### Custom Allergens")

    input_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")

    with input_col: 
        # Saves what the user types in the text box so the website doesn't forget it
        st.text_input("Add any allergens not mentioned above", key="custom_allergen_input")

    with btn_col: 
        if st.button("Add", use_container_width=True):
            # Adds the custom typed out allergen to the list and turns its switch on 
            new_al = st.session_state.custom_allergen_input.strip().title()
            
            if new_al and new_al not in st.session_state.allergen_list:
                st.session_state.allergen_list.append(new_al)
                
                # Makes sure that custom added allergen is in the active list 
                if new_al not in st.session_state.active_allergens:
                    st.session_state.active_allergens.append(new_al)
                st.rerun()

def clean_text(text):
    text = re.sub(r"\$?\d+(\.\d+)?", "", text)
    text = re.sub(r"[^a-zA-Z& ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

JUNK_WORDS = [
    "DELICIOUS FOOD", "MENU", "APPETIZER", "APPETIZERS",
    "SALAD & SOUP", "SALAD SOUP", "DRINKS", "D RINKS", "FOOD"
]
# SCANNER PAGE SETUP
def scan():
    if "allergen_list" not in st.session_state:
        st.session_state.allergen_list = ['Milk', 'Wheat', 'Tree Nut', 'Soy', 'Fish', 'Shellfish', 'Gluten', 'Egg']
    if "active_allergens" not in st.session_state:
        st.session_state.active_allergens = [] 
    
    st.markdown("""
        <div style="text-align: center;">
            <h1 class="title-box">Scan Your Menu</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload or capture a menu to check for your active allergens.</p>", unsafe_allow_html=True)
    st.write("---")

    op1, op2 = st.tabs(["Upload a File", "Use Camera"])

    # FILE UPLOADING TAB ---
    with op1: 
        st.write("#### Select a file from your device")
        upload_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"], key="file_uploader")

        if upload_file is not None: 
            st.success("Image uploaded")
            st.image(upload_file, caption="Preview", use_container_width=True)
            
            if st.button("Analyzing for Allergens", type="primary", use_container_width=True, key="btn_upload"):
                with st.spinner("Reading menu and running ML model..."):
                    
                    # Prepares the uploaded image so the text reader can understand it
                    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)

                    # Scans the image for words, filters out the junk, and then make a list of the food items
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    reader = easyocr.Reader(["en"])
                    results = reader.readtext(blurred)

                    foodItems = []
                    for _, text, conf in results:
                        if conf > 0.5:
                            cleaned = clean_text(text)
                            if len(cleaned) > 2 and cleaned.upper() not in JUNK_WORDS:
                                foodItems.append(cleaned)

                    # Sends the food list to the ML model to guess what ingredients and allergens are present
                    ml_results = predict_allergens(foodItems)

                    # Displays the results 
                    st.write("---")
                    st.subheader("Analysis Results")
                    
                    # Takes the user's allergens from the profile for matching
                    active_allergens = [a.lower() for a in st.session_state.active_allergens]                    

                    if not ml_results:
                        st.info("No food items detected.")
                    
                    for r in ml_results:
                        food_name = r["food_item"]
                        detected_allergens = r["predicted_allergens"]
                        
                        # Makes all text lowercase so 'Wheat' and 'wheat' will match 
                        # Cleans the output of the ML model for safe matching
                        food_name_lower = food_name.lower()
                        detected_allergens_lower = [a.lower() for a in detected_allergens]
                        matched_ingredients_lower = [i.lower() for i in r['matched_ingredients']]

                        is_dangerous = False
                        
                        # Scans the food's name and ingredients matches the user's allergens in the active list
                        for active in active_allergens:
                            if (active in detected_allergens_lower) or \
                               (active in matched_ingredients_lower) or \
                               (active in food_name_lower):
                                is_dangerous = True
                                break # No need to keep looking once we know it's unsafe

                        # Shows a warning if the food is dangerous, or puts it in a green box if it's safe 
                        if is_dangerous:
                            st.error(f"🚨 **WARNING: {food_name}**")
                            st.write(f"**Contains:** {', '.join(detected_allergens)}")
                            st.write(f"**Matched Ingredients:** {', '.join(r['matched_ingredients'])}")
                            st.caption(f"Confidence: {r['confidence']:.1%}")
                        else:
                            with st.expander(f"✅ {food_name} (Safe)"):
                                st.write(f"**Contains:** {', '.join(detected_allergens) if detected_allergens else 'None detected'}")
                                st.write(f"**Matched Ingredients:** {', '.join(r['matched_ingredients']) if r['matched_ingredients'] else 'N/A'}")
                                st.caption(f"Confidence: {r['confidence']:.1%}")

    # CAMERA TAB ---
    with op2: 
        st.write("#### Capture a photo")
        camera = st.camera_input("Take a picture")

        if camera is not None: 
            st.success("Photo captured")
            
            if st.button("Analyzing for Allergens", type="primary", use_container_width=True, key="btn_camera"):
                with st.spinner("Reading menu and running ML model..."):
                    
                    # Prepares the camera photo so the text reader can understand it
                    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)

                    # Scans the image for words, filters out the junk, and then make a list of the food items
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    reader = easyocr.Reader(["en"])
                    results = reader.readtext(blurred)

                    foodItems = []
                    for _, text, conf in results:
                        if conf > 0.5:
                            cleaned = clean_text(text)
                            if len(cleaned) > 2 and cleaned.upper() not in JUNK_WORDS:
                                foodItems.append(cleaned)

                    # Sends the food list to the ML model to guess what ingredients and allergens are present
                    ml_results = predict_allergens(foodItems)

                    # Displays the results
                    st.write("---")
                    st.subheader("Analysis Results")
                    
                    # Takes the active list so we can check the allergens that are toggled "on"
                    user_allergens = [a.lower() for a in st.session_state.allergen_list]

                    if not ml_results:
                        st.info("No food items detected.")
                    
                    for r in ml_results:
                        food_name = r["food_item"]
                        detected_allergens = r["predicted_allergens"]
                        
                        is_dangerous = any(allergen.lower() in user_allergens for allergen in detected_allergens)

                        # Shows a warning if the food is dangerous, or puts it in a green box if it's safe 
                        if is_dangerous:
                            st.error(f"🚨 **WARNING: {food_name}**")
                            st.write(f"**Contains:** {', '.join(detected_allergens)}")
                            st.write(f"**Matched Ingredients:** {', '.join(r['matched_ingredients'])}")
                            st.caption(f"Confidence: {r['confidence']:.1%}")
                        else:
                            with st.expander(f"✅ {food_name} (Safe)"):
                                st.write(f"**Contains:** {', '.join(detected_allergens) if detected_allergens else 'None detected'}")
                                st.write(f"**Matched Ingredients:** {', '.join(r['matched_ingredients']) if r['matched_ingredients'] else 'N/A'}")
                                st.caption(f"Confidence: {r['confidence']:.1%}")


home = st.Page(homepage, title="HomePage",  default = True)
prof = st.Page(profile, title = "Profile")
scanner = st.Page(scan, title = "Scanner")

pg = st.navigation([home,prof,scanner])

pg.run()
