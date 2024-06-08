import streamlit as st
import pandas as pd
import pickle 

st.set_page_config(layout="wide", page_title="Mushroom Classifier: Poisonous or Edible")
with open('./model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
feature_input = dict()

st.markdown("<h2 style='text-align: center; color: white;'>Mushroom Classifier: Poisonous or Edible</h1>", unsafe_allow_html=True)
colA, colB = st.columns(spec=[10, 0.5])
tabA, tabB = st.tabs(['About', 'Run model'])

with tabA:
    df = pd.read_csv('test_data.csv')
    st.write("This project builds a classification model that helps determine if a \
             mushroom is dangerous to eat based on its physical characteristics.") 
    st.write(df)
    st.write("The table above shows the test split of the data and is provided to \
             visualize prediction accuracy with the class feature.")
    st.write("If the mushroom is classified under a 0, then it is edible and safe for consumption.\
             But, if the mushroom is classified under a 1, then it is poisonous and should not be ingested.")    

with tabB:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        cap_shape   = st.radio('**Cap Shape**',   ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'])
        cap_surface = st.radio('**Cap Surface**', ['fibrous', 'grooves', 'scaly', 'smooth'])
        cap_color   = st.radio('**Cap Color**',   ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'])
    with col2:
        gill_spacing = st.radio('**Gill Spacing**', ['close', 'crowded', 'distant'])
        gill_size    = st.radio('**Gill Size**',    ['broad', 'narrow'])     
        gill_color   = st.radio('**Gill Color**',   ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'])    
    with col3:
        stalk_surface_above_ring = st.radio('**Stalk Surface Above Ring**', ['fibrous', 'scaly', 'silky', 'smooth'])
        stalk_color_above_ring   = st.radio('**Stalk Color Above Ring**',   ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
        ring_type                = st.radio('**Ring Type**',                ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'])
    with col4:
        bruises           = st.radio('**Bruises**',           ['bruises', 'no'])
        odor              = st.radio('**Odor**',              ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'])
        spore_print_color = st.radio('**Spore Print Color**', ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'])
    with col5:
        population = st.radio('**Population**', ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'])
        habitat    = st.radio('**Habitat**',    ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'])

    feature_input['cap shape']   = cap_shape
    feature_input['cap surface'] = cap_surface
    feature_input['cap color']   = cap_color

    feature_input['gill spacing'] = gill_spacing
    feature_input['gill size']    = gill_size
    feature_input['gill color']   = gill_color

    feature_input['stalk surface above ring'] = stalk_surface_above_ring
    feature_input['stalk color above ring']   = stalk_color_above_ring
    feature_input['ring type']                = ring_type

    feature_input['bruises']           = bruises
    feature_input['odor']              = odor
    feature_input['spore print color'] = spore_print_color

    feature_input['population'] = population
    feature_input['habitat']    = habitat

    st.write("After specifying the mushroom's features, press the button below to determine its edibility.")
    if st.button("**Predict**"):
        X_test     = dv.transform(feature_input)
        prediction = model.predict(X_test)
        result     = prediction[0]
        if result == 0.0:
            st.markdown('The mushroom is **edible**.')
        else:
            st.markdown('The mushroom is **poisonous**.')

