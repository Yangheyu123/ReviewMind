import pytest

from app.schemas.review import ConversationRole, ConversationTurn
from app.services.conversation_store import conversation_store


@pytest.mark.anyio
async def test_append_and_list_turns_in_order():
    t1 = await conversation_store.append(
        ConversationTurn(
            job_id="rev_c1",
            finding_id="sec_1",
            role=ConversationRole.user,
            content="这是误报",
        )
    )
    t2 = await conversation_store.append(
        ConversationTurn(
            job_id="rev_c1",
            finding_id="sec_1",
            role=ConversationRole.assistant,
            content="已撤销",
        )
    )

    turns = await conversation_store.list("rev_c1", finding_id="sec_1")
    assert len(turns) == 2
    assert turns[0].role == ConversationRole.user
    assert turns[1].role == ConversationRole.assistant
    assert turns[0].content == "这是误报"
    assert t1.created_at is not None
    assert t2.created_at is not None


@pytest.mark.anyio
async def test_list_filters_by_finding_id():
    await conversation_store.append(
        ConversationTurn(
            job_id="rev_c2",
            finding_id="f_a",
            role=ConversationRole.user,
            content="a",
        )
    )
    await conversation_store.append(
        ConversationTurn(
            job_id="rev_c2",
            finding_id="f_b",
            role=ConversationRole.user,
            content="b",
        )
    )

    only_a = await conversation_store.list("rev_c2", finding_id="f_a")
    assert len(only_a) == 1
    assert only_a[0].content == "a"

    all_turns = await conversation_store.list("rev_c2")
    assert len(all_turns) == 2


@pytest.mark.anyio
async def test_list_empty_for_unknown_job():
    assert await conversation_store.list("rev_nonexistent") == []
