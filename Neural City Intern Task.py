import streamlit as st
import pandas as pd
import contextlib
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Gov Data Analyst", layout="centered")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Python data analyst. You have a pandas DataFrame named `df`. Schema:\n{schema}\nWrite pure pandas code to answer the user's question. Save the final output to a variable named `result`. Return only raw code without markdown backticks. If the question cannot be answered with this schema, return exactly: OUT_OF_SCOPE"),
    ("user", "{question}")
])

synth_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a single clear sentence answering the user's question based on the raw data result."),
    ("user", "Question: {question}\nResult: {raw_result}")
])

def run_code(code, df):
    clean_code = code.replace("```python", "").replace("```", "").strip()
    if clean_code == "OUT_OF_SCOPE":
        return None, "This question is out of scope for the current dataset."
    
    local_env = {"df": df, "pd": pd}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(clean_code, {}, local_env)
        
        if "result" not in local_env:
            return None, "The code executed but did not assign a value to the 'result' variable."
            
        return local_env.get("result"), clean_code
    except Exception as e:
        return None, f"Execution Error: {e}\n\nAttempted Code:\n{clean_code}"

st.title("Talk to Government Data")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    with st.expander("Preview Data"):
        st.dataframe(df.head())
        
    query = st.text_input("Ask a question about this data:")
    
    if st.button("Analyze") and query:
        with st.spinner("Processing..."):
            schema = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
            
            chain = code_prompt | llm | StrOutputParser()
            code_str = chain.invoke({"schema": schema, "question": query})
            
            raw_data, context = run_code(code_str, df)
            
            if raw_data is None:
                st.warning(context)
            else:
                synth_chain = synth_prompt | llm | StrOutputParser()
                answer = synth_chain.invoke({"question": query, "raw_result": str(raw_data)})
                
                st.success(answer)
                
                if isinstance(raw_data, (pd.DataFrame, pd.Series)):
                    st.dataframe(raw_data)
                    if isinstance(raw_data, pd.Series) and pd.api.types.is_numeric_dtype(raw_data):
                        st.bar_chart(raw_data)
                
                with st.expander("Audit Query"):
                    st.code(context, language="python")
