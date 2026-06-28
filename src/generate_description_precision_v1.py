from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "outputs" / "submission_backup_3825_before_desc_precision.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "outputs" / "reports"


DESCRIPTION_UPDATES = {
    366: (
        "Gladius swap execution lets filler-provided executor data determine the actual swap route while the order "
        "only constrains final outputs. A filler can route through worse liquidity or stale parameters and keep the "
        "surplus value that should be protected by the maker's minimum output."
    ),
    367: (
        "Gladius reactor callbacks accept externally supplied execution data without fully binding the callback path "
        "to the signed order intent. A malformed executor payload can make the order settle under token-flow "
        "assumptions that the maker did not authorize."
    ),
    368: (
        "M^0 delegation checkpoints can keep expired or stale voting relationships active when epoch state is not "
        "validated before delegation changes. This lets accounts reuse old voting influence and distorts proposal "
        "weight calculations."
    ),
    369: (
        "Signature-based governance actions in M^0 require nonce and expiry binding for every accepted digest. When "
        "nonce consumption is inconsistent, the same signed authorization can be replayed to repeat a vote or "
        "delegation action."
    ),
    370: (
        "The dnGmx vault uses fixed slippage thresholds while converting USDC, GLP, GMX, ETH and BTC exposure. During "
        "volatile markets those static bounds allow deposits or withdrawals to clear at adverse prices, shifting "
        "losses to vault users."
    ),
    371: (
        "DnGmx vault accounting prices GLP and hedge exposure from Aave and GMX oracle data. Accepting stale or "
        "manipulated prices during rebalance or withdrawal misvalues shares and lets users receive more or less than "
        "their fair portion of assets."
    ),
    372: (
        "DODO V3 relies on Chainlink-style feeds to value pools before liquidation. Without checking stale rounds, "
        "invalid answers and sequencer downtime, liquidation decisions can use unsafe prices and seize collateral at "
        "the wrong valuation."
    ),
    374: (
        "Gladius signed orders depend on nonce invalidation to prevent duplicate fills. If nonce state is not bound "
        "to each executable order variant, a previously signed order can be filled again through a different reactor "
        "or executor path."
    ),
    375: (
        "The Rubicon reactor integrates arbitrary ERC20 tokens through Permit2 and swap callbacks. Non-standard "
        "tokens that do not return expected transfer values can make reactor accounting record a fill even when the "
        "contract receives fewer tokens than required."
    ),
    378: (
        "M^0 registrar permissions control minters, validators, earners and governance modules. Stale or overly broad "
        "roles let privileged accounts change protocol-critical approvals and bypass the intended minting or earning "
        "authorization process."
    ),
    381: (
        "DnGmxJuniorVault uses flashloan state to coordinate leveraged hedge operations. If the callback does not "
        "verify repayment and expected post-loan balances, the vault can finish with inconsistent debt, hedge, or "
        "GLP accounting."
    ),
    384: (
        "Rage Trade vault health depends on timely rebalancing when GLP and hedge exposure move. When liquidation or "
        "rebalance rewards do not cover gas and slippage, keepers have no incentive to fix unhealthy positions and "
        "losses remain in the vault."
    ),
    385: (
        "D3VaultLiquidation lets liquidators specify debtToCover and collateralAmount. If the vault does not derive "
        "these values from current debt and collateral balances, a liquidator can repay too little debt and receive "
        "too much collateral."
    ),
    386: (
        "DODO V3 pool removal passes through pending liquidation and repayment states. Missing status validation lets "
        "repayment or removal functions run while a pool is pending removal, causing inconsistent accounting and "
        "blocked cleanup."
    ),
    388: (
        "D3MMLiquidationRouter performs swaps while settling liquidations. Without a strict minimum output check for "
        "the liquidation path, collateral can be swapped at an unfavorable price and reduce the value recovered by "
        "the vault."
    ),
    389: (
        "RubiconFeeController can set base and pair-specific fees that are deducted from reactor outputs. If those "
        "fees are not capped, the controller can configure excessive fees and capture most of a user's swap output."
    ),
    391: (
        "Arrakis module changes move liquidity while fees and reserves are still being accounted for. If reserve "
        "state is not synchronized before setModule completes, vault share pricing diverges from the actual liquidity "
        "held by the active module."
    ),
    392: (
        "MinterGateway calculates minting capacity from collateral, validator approvals and registrar state. Weak "
        "bounds on collateral or debt inputs let a minter overstate backing assets and mint more M tokens than the "
        "position can safely support."
    ),
    393: (
        "M^0 voting power is snapshotted by epoch, but delegation updates around epoch boundaries can change the "
        "effective delegate after proposal weight should be fixed. This bypasses the intended governance snapshot "
        "semantics."
    ),
    394: (
        "M^0 EIP712 helpers must reject expired signatures and consume nonces atomically with each accepted action. "
        "Separating validation from nonce updates allows the same signed message to authorize repeated delegation or "
        "governance operations."
    ),
    396: (
        "DnGmx BatchingManager settles deposits and withdrawals by round. If failed GLP staking or partial settlement "
        "does not advance round state correctly, users remain assigned to an old round and cannot complete their "
        "vault operation."
    ),
    398: (
        "D3Oracle consumes Chainlink and sequencer feeds for token valuation. Stale answers or sequencer downtime "
        "must be rejected before liquidation checks, otherwise pools are liquidated or protected using outdated "
        "prices."
    ),
    399: (
        "D3VaultLiquidation should recompute debtToCover and collateralAmount from live pool balances. Trusting user "
        "provided values allows a liquidation to claim excessive collateral for insufficient repayment."
    ),
    400: (
        "DODO V3 pool removal requires liquidation and pending repayment to finish before final removal. Deprecated "
        "or pending-remove pools that are not handled by all liquidation paths can become stuck and block pool "
        "cleanup."
    ),
}


def clean(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(BASE_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    if list(sub.columns) != list(sample.columns):
        raise ValueError("Base columns do not match submission example columns")

    old_rows = []
    for prop, desc in DESCRIPTION_UPDATES.items():
        idx = sub.index[sub["Property"].astype(int) == prop]
        if len(idx) != 1:
            raise ValueError(f"Property {prop} not found exactly once")
        i = idx[0]
        old_rows.append((prop, sub.at[i, "repo_path"], sub.at[i, "description"], desc))
        sub.at[i, "description"] = clean(desc)

    output_path = OUT_DIR / "submission_description_precision_v1.csv"
    sub.to_csv(output_path, index=False)
    shutil.copy2(output_path, ROOT / "submission.csv")

    report_lines = [
        "# Description Precision V1 Report",
        "",
        f"- Base: `{BASE_PATH}`",
        f"- Output: `{output_path}`",
        f"- Descriptions changed: `{len(DESCRIPTION_UPDATES)}`",
        "- Rows, repo_path, severity, tag and subtag are preserved.",
        "",
        "## Updated Properties",
    ]
    for prop, repo, old, new in old_rows:
        report_lines.append(f"- `{prop}` `{repo}`: {len(clean(old))} chars -> {len(clean(new))} chars")
    (REPORT_DIR / "description_precision_v1_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}")
    print("Copied to submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
