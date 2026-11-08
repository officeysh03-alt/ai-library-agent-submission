"""AI 사서 보조 도구. 수업 05_full_agent.py의 @tool/Context 구조를 적용.

Ollama 수정 초안을 감독 검토하여 보완한 3파트 모듈.
가드레일·기억·최종 검수·대화 실행기는 4파트에서 통합한다.
"""
from dataclasses import dataclass
from typing import Dict, Any
from langchain.tools import tool, ToolRuntime
import copy

# 모델 상수
MODEL = "qwen3:4b-instruct-2507-q4_K_M"


@dataclass(frozen=True)
class Context:
    """실행 프로그램이 지정하는 신원. 채팅이나 모델 도구 인자로 변경하지 않는다."""
    librarian_name: str
    librarian_role: str
    user_id: str
    app_name: str
    demo_mode: bool = False

# 도서 목록 (정확히 11권)
books = [
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]

@tool
def search_books(query: str) -> Dict[str, Any]:
    """
    도서 제목이나 저자 이름으로 책을 검색합니다.
    공백을 정리하고, 대소문자를 무시하여 검색합니다.
    검색 결과가 없으면 'not_found'를 반환합니다.
    공백 질의는 'invalid_query'로 처리합니다.
    """
    if not query or not query.strip():
        return {"status": "invalid_query", "message": "검색어가 비어 있습니다.", "books": []}
    
    query = " ".join(query.split()).casefold()
    results = []
    for book in books:
        if query in book["title"].casefold() or query in book["author"].casefold():
            results.append(copy.deepcopy(book))
    
    if not results:
        return {"status": "not_found", "message": "도서를 찾을 수 없습니다.", "books": []}
    
    return {"status": "ok", "message": f"검색 결과: {len(results)}권", "books": results}

@tool
def check_my_loans(runtime: ToolRuntime[Context]) -> Dict[str, Any]:
    """
    사용자의 대출 정보를 확인합니다.
    runtime.context에서 user_id를 읽어 사용자 정보를 추출합니다.
    원장이 없으면 unavailable, demo_mode일 때 가상 원장 사용.
    사용자가 존재하지 않으면 unavailable.
    모든 데모 응답은 fictional=True로 표시합니다.
    """
    context = runtime.context
    user_id = context.user_id
    fictional = context.demo_mode is True
    
    if not user_id:
        return {"status": "unavailable", "message": "현재 이용자 신원을 확인할 수 없습니다.", "loans": None, "fictional": fictional}
    
    # demo_mode일 때 가상 원장 사용
    demo_mode = fictional
    if demo_mode:
        if user_id == "demo_reader_a":
            return {"status": "ok", "message": "[가상 실험 데이터] 본인 대출 1권입니다. 실제 대출 정보가 아닙니다.", "loans": [copy.deepcopy(books[0])], "fictional": True}
        elif user_id == "demo_reader_b":
            return {"status": "ok", "message": "[가상 실험 데이터] 본인 대출 0건으로 확인됩니다. 실제 대출 정보가 아닙니다.", "loans": [], "fictional": True}
        else:
            return {"status": "unavailable", "message": "[가상 실험 데이터] 현재 이용자의 원장이 없어 대출 정보를 확인할 수 없습니다.", "loans": None, "fictional": True}
    
    # 실제 원장 없음 → unavailable
    return {"status": "unavailable", "message": "대출 원장이 제공되지 않아 대출 정보를 확인할 수 없습니다. 대출 내역이 없다는 의미는 아닙니다.", "loans": None, "fictional": False}

@tool
def reply_to_user(body: str) -> Dict[str, Any]:
    """
    이용자에게 응답을 작성합니다.
    본문이 비어 있으면 invalid_body로 거절.
    응답은 draft 상태로 반환되며, sent=False로 설정.
    """
    if not body or not body.strip():
        return {"status": "invalid_body", "message": "응답 본문이 비어 있습니다.", "body": "", "sent": False}
    
    return {"status": "draft", "message": "검수 대기 중인 회신 초안입니다. 실제 발송하지 않았습니다.", "body": body, "sent": False}
