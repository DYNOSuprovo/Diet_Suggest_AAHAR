from langchain_core.callbacks.base import BaseCallbackHandler

class SafeTracer(BaseCallbackHandler):
    def on_chain_end(self, outputs: dict, **kwargs):
        try:
            if isinstance(outputs, dict) and "answer" in outputs:
                print("🔁 Chain ended. Answer:", outputs["answer"])
            else:
                print("🔁 Chain ended. Output (no 'answer' key):", outputs)
        except Exception as e:
            print("❌ Error in on_chain_end callback:", e)
