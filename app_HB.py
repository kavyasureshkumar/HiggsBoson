
import streamlit as st
import tensorflow as tf
import numpy as np

def predict(model, inp):
    y = model.predict(inp)
    y_pred = y.argmax(axis = 1)
    if y_pred == 0:
        return 'b'
    else:
        return 's'

model = tf.keras.models.load_model('Higgs_Boson.h5')


features = ['DER_mass_transverse_met_lep', 'DER_pt_h', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt', 'PRI_met_sumet', 'PRI_jet_all_pt', 'Weight']
imputed_values = [0, 0, 225.885, -0.244, 0, 0, 0, 0, 0, 0]


def main():
    st.header('Higgs Boson Event Detection')

    st.write('The model was trained on the Higgs Boson dataset')
    st.write('Labels: \'s\' : Signal, \'b\' : Background')

    st.write('Enter the required data below')

    m_t = st.number_input(features[0])
    pt_h = st.number_input(features[1])
    m_j_j = st.number_input(features[2])
    prodeta = st.number_input(features[3])
    ratio_lep_tau = st.number_input(features[4])
    met_phi_cen = st.number_input(features[5])
    tau_pt = st.number_input(features[6])
    met_sumer = st.number_input(features[7])
    jet_all_pt = st.number_input(features[8])
    weight = st.number_input(features[9])

    if m_j_j == -999:
        m_j_j = imputed_values[2]
    if prodeta == -999:
        prodeta = imputed_values[3]
    input = np.array([[m_t,pt_h,m_j_j,prodeta,ratio_lep_tau,met_phi_cen,tau_pt,met_sumer,jet_all_pt,weight]])
    print(input)
    if st.button('Predict Event'):
        pred = predict(model,input)        
        st.success('The event predicted by the model is ' + pred)

if __name__ == '__main__':
    main()
