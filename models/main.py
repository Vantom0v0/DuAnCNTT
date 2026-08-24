from naive_rag import naive_rag_model
from iter_retgen import iter_retgen_model
from ircot import ircot_model
from rag_fusion import rag_fusion_model
from s2g_rag import s2g_rag_model

if __name__ == "__main__":
    query = "Em khâu sụn chêm gối phải được 4 tuần. Bác sĩ cho em hỏisau khâu sụn chêm gối 4 tuần có thể bỏ nẹp đi lại chưa?Em đi thử không đau nhưng vẫn hơi tập tễnh. Em gập được 100 độ. Em cảm ơn bác sĩ."

    answer = s2g_rag_model(query)