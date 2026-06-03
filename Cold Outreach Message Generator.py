import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def generate_outreach_content(business_desc: str, outreach_type: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.8,
            api_key=api_key
        )

    templates = {
        "WhatsApp Message": """
            You are the owner/founder of the business described below. You are writing a direct WhatsApp outreach message FROM your business TO potential corporate clients, local partners, or event managers to get more customers.
            
            Your Business Context: "{business_description}"
            
            Task: Write a short outreach message acting AS this business.
            
            Strict Anti-AI & Humanized Rules:
            - ABSOLUTELY FORBIDDEN phrases: "Look no further!", "Are you tired of...", "Revolutionize your...", "In today's fast-paced world", "Introducing...", "Dear [Name]".
            - Do not sound like a marketing agency pitching to this business. You ARE this business pitching to others.
            - Keep it casual, direct, and under 4-5 short lines. Use generous line breaks.
            - Max 1 normal emoji (like 👋 or ☕) or none. No bullet points.
            - Start with a direct, human greeting like "Hey [Name]," or "Hi [Name] -".
            - State who you are locally and pitch a simple, valuable offer (e.g., corporate catering packages, free team trials, or bulk local delivery).
            - End with a zero-pressure, conversational question like "Worth a quick chat?" or "Can I drop off a few samples for your team next week?"
            - Output ONLY the raw message. No preamble.
        """
    }

    prompt = PromptTemplate(
        input_variables=["business_description"],
        template=templates[outreach_type]
    )

    chain = prompt | llm | StrOutputParser()


    return chain.invoke({"business_description": business_desc})
   

def main():
    st.set_page_config(
        page_title="AI Outreach Hub", 
        page_icon="🚀", 
        layout="centered"
    )

    st.title("🚀 Custom Outreach Engine")
    st.markdown("Transform business profiles into raw, human-sounding outreach assets.")
    st.write("---")

    business_description = st.text_area(
        "📝 Business Profile & Description", 
        placeholder="Provide details about the business model and target audience...",
        height=160
    )

    tab_whatsapp, = st.tabs(["WhatsApp Channel"])

    with tab_whatsapp:
        st.subheader("WhatsApp Campaign Copy")
        st.caption("Humanized, short-form messaging stripped of marketing buzzwords.")
        if st.button("Generate WhatsApp Copy", type="primary", key="btn_wa"):
            if not business_description.strip():
                st.warning("Please provide a business profile description before running execution.")
            else:
                with st.spinner("Stripping out marketing filler..."):
                    result = generate_outreach_content(business_description, "WhatsApp Message")
                    st.success("Generation Complete!")
                    st.text_area("Copy-Paste Ready Version:", value=result, height=300)


if __name__ == "__main__":
    main()
