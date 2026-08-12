import json
from pathlib import Path

from my_agent.core.delivery import (
    DeliveryVerification,
    aggregate_delivery_result,
    verify_project_delivery,
)


def test_delivery_result_aggregates_successful_write_and_failed_build() -> None:
    result = aggregate_delivery_result(
        context_status="failed",
        context_reason="delivery_verification_failed: npm run build failed",
        verification=DeliveryVerification(
            checked=True,
            passed=False,
            failures=["npm run build failed: missing component"],
        ),
        manifest={"created": ["src/App.vue"], "modified": ["src/game/engine.js"]},
    )

    assert result.write_status == "success"
    assert result.build_status == "failed"
    assert result.final_status == "not_accepted"
    assert result.changed_files == ["src/App.vue", "src/game/engine.js"]


async def test_delivery_verification_catches_missing_vue_component(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.vue").write_text(
        "<script setup>\nimport TankGame from './components/TankGame.vue'\n</script>\n",
        encoding="utf-8",
    )
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"build": "vite build"}}),
        encoding="utf-8",
    )

    result = await verify_project_delivery(tmp_path, "创建一个 Vite Vue 项目")

    assert result.checked
    assert not result.passed
    assert "src/App.vue imports missing ./components/TankGame.vue" in result.failures


async def test_delivery_verification_passes_resolved_import_without_build_script(
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "components").mkdir(parents=True)
    (tmp_path / "src" / "App.vue").write_text(
        "<script setup>\nimport TankGame from './components/TankGame.vue'\n</script>\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "components" / "TankGame.vue").write_text(
        "<template><div>tank</div></template>\n", encoding="utf-8"
    )

    result = await verify_project_delivery(tmp_path, "create a Vue project")

    assert result.checked
    assert result.passed
