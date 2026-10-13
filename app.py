import streamlit as st
import pandas as pd

st.set_page_config(page_title='Solar AI Monitor', page_icon='☀️', layout='wide')

st.title('☀️ Solar AI Monitor')
st.subheader('AI-Powered Solar Panel Performance Monitoring')
st.write('Explore how environmental conditions may influence solar panel performance.')

st.sidebar.header('Environmental Conditions')
temperature = st.sidebar.slider('Panel temperature (°C)', 10, 90, 40)
dust = st.sidebar.slider('Dust accumulation (%)', 0, 100, 20)
humidity = st.sidebar.slider('Humidity (%)', 0, 100, 40)

# Illustrative heuristic only; this is not a trained AI model.
temperature_impact = max(0, (temperature - 25) * 0.35)
dust_impact = dust * 0.25
humidity_impact = humidity * 0.05
estimated_loss = min(95, temperature_impact + dust_impact + humidity_impact)
risk = 'High' if estimated_loss >= 25 else 'Moderate' if estimated_loss >= 12 else 'Low'

st.divider()
st.markdown('### Performance Overview')
c1, c2, c3 = st.columns(3)
c1.metric('Panel Temperature', f'{temperature} °C')
c2.metric('Illustrative Estimated Loss', f'{estimated_loss:.1f}%')
c3.metric('Estimated Risk Level', risk)

st.markdown('### Environmental Impact Analysis')
impact_data = pd.DataFrame({'Estimated contribution (%)': [temperature_impact, dust_impact, humidity_impact]}, index=['Temperature', 'Dust', 'Humidity'])
st.bar_chart(impact_data)
st.caption('Bars show illustrative contributions before the overall estimate cap. They are not measured losses.')

st.markdown('### Recommended Actions')
if temperature >= 60:
    st.warning('Temperature is elevated. Check ventilation and the panel manufacturer specifications.')
else:
    st.info('Monitor panel temperature and compare it with the manufacturer operating range.')
if dust >= 50:
    st.warning('Dust level is high. Inspect panel cleanliness and consider safe cleaning if needed.')
else:
    st.info('Inspect panel surfaces periodically, especially after dusty weather.')
if humidity >= 80:
    st.warning('Humidity is high. Check for moisture-related issues and follow manufacturer guidance.')
if estimated_loss >= 25:
    st.error('The selected conditions indicate high estimated risk in this illustrative model. Verify with real production data.')
elif estimated_loss >= 12:
    st.warning('The selected conditions indicate moderate estimated risk. Compare actual output with expected production.')
else:
    st.success('The selected conditions indicate low estimated risk in this illustrative model.')

with st.expander('Methodology and Limitations'):
    st.write('This prototype uses a manually defined formula to illustrate possible environmental impacts. It does not contain a trained AI model, and its estimates are not validated predictions.')
    st.write('Reliable forecasting requires historical solar power output, weather and panel-condition data, a trained model, and evaluation on unseen data.')

st.caption('Research prototype • Estimates are illustrative, not measured or validated predictions.')
