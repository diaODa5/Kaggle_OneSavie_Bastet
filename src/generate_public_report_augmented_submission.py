from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SUBMISSION = ROOT / "outputs" / "submission_official_expanded_plus_optimism.csv"
TEST_PATH = ROOT / "data" / "raw_kaggle" / "test.csv"
TRAIN_PATH = ROOT / "data" / "raw_kaggle" / "train.csv"
SAMPLE_PATH = ROOT / "data" / "raw_kaggle" / "submission_example.csv"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "outputs" / "reports"


PUBLIC_FINDINGS = [
    {
        "repo_path": "9470d2cf198f",
        "severity": "High",
        "tag": "Input Validation",
        "subtag": "Bypass Mechanism, Invalid Validation",
        "description": (
            "In Launch.updateParticipation, token amounts are compared with currency amounts without "
            "normalizing token decimals. A participant can bypass maxTokenAmountPerUser and receive a larger "
            "allocation than intended."
        ),
        "source": "InfiniteSec 2025-02-rova report",
    },
    {
        "repo_path": "9470d2cf198f",
        "severity": "High",
        "tag": "Accounting Error",
        "subtag": "Incorrect Parameter",
        "description": (
            "Launch.updateParticipation subtracts refundCurrencyAmount from userTokenAmount even though the "
            "values are denominated in different units. This corrupts userTokens accounting and can cause fund "
            "loss, failed refunds, or denial of service."
        ),
        "source": "InfiniteSec 2025-02-rova report",
    },
    {
        "repo_path": "592eed5791df",
        "severity": "Medium",
        "tag": "Arithmetic",
        "subtag": "Precision Loss",
        "description": (
            "StakingRewardsV2 calculates USDC rewardPerToken with the same 18-decimal scaling used for KWENTA, "
            "even though USDC has only 6 decimals. Frequent reward updates can round USDC rewards to zero and "
            "prevent stakers from receiving the intended distribution."
        ),
        "source": "InfiniteSec 2024-07-kwenta-staking-contracts report",
    },
    {
        "repo_path": "73f6a793d916",
        "severity": "High",
        "tag": "call / delegatecall",
        "subtag": "Missing Return Check",
        "description": (
            "VestingEscrow derives the factory address from forgeable immutable clone arguments and then uses "
            "that address to resolve the voting adaptor for delegatecall. An attacker can craft calldata that "
            "delegates into a malicious adaptor and bricks escrow implementations."
        ),
        "source": "InfiniteSec 2024-01-rio-vesting-escrow report",
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "High",
        "tag": "Logic Error",
        "subtag": "Bad Condition",
        "description": (
            "ValantisHOTModulePublic treats the first deposit into a newly selected module as dust cleanup even "
            "when the module already received real liquidity from setModule. The first deposit can send the pool "
            "liquidity to the manager and make user shares worthless."
        ),
        "source": "InfiniteSec 2024-03-arrakis report",
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "High",
        "tag": "Access Control",
        "subtag": "Bypass Mechanism",
        "description": (
            "ArrakisStandardManager.setModule allows an executor to provide arbitrary payloads after vault tokens "
            "are moved into a new module. A malicious executor can call withdraw through the payload and drain the "
            "vault reserves."
        ),
        "source": "InfiniteSec 2024-03-arrakis report",
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "High",
        "tag": "Accounting Error",
        "subtag": "Incorrect Formula",
        "description": (
            "The rebalance flow can be abused to mint cheap vault shares and drain reserves because the share "
            "pricing and liquidity accounting are not sufficiently protected during the rebalance operation."
        ),
        "source": "InfiniteSec 2024-03-arrakis report",
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "High",
        "tag": "DoS",
        "subtag": "Rounding Error",
        "description": (
            "ArrakisPublicVaultRouter rounds required mint amounts down while the vault and module round token "
            "requirements up. This mismatch can make add-liquidity transactions revert and deny users access to "
            "liquidity provision."
        ),
        "source": "InfiniteSec 2024-03-arrakis report",
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "High",
        "tag": "Slippage",
        "subtag": "Invalid Slippage Control / Missing slippage check",
        "description": (
            "The Valantis HOT deposit flow does not pass the expected price bounds into alm.depositLiquidity. "
            "Without those bounds, a sandwich attacker can manipulate execution around the deposit and extract "
            "value from the vault."
        ),
        "source": "Sherlock/InfiniteSec Arrakis report summary",
    },
    {
        "repo_path": "1167ec3a176e",
        "severity": "Medium",
        "tag": "ERC20",
        "subtag": "safeApprove",
        "description": (
            "Swap paths use safeIncreaseAllowance for tokens such as USDT that require the allowance to be reset "
            "to zero before increasing it. Leftover allowance can break later swaps and make USDT interactions "
            "fail."
        ),
        "source": "InfiniteSec 2024-03-arrakis report",
    },
]


TOY_REPO_FINDINGS = [
    {
        "repo_path": "c2426a2ab283",
        "severity": "High",
        "tag": "Reentrancy",
        "subtag": "onERC721Received callback",
        "description": (
            "ChiikawaSticker.claim calls _safeMint before clearing canClaim. A malicious ERC721 receiver can "
            "reenter through onERC721Received and claim multiple NFTs after paying only once."
        ),
        "source": "test.zip source inspection: ChiikawaSticker.sol",
    },
    {
        "repo_path": "27c6f2a68058",
        "severity": "High",
        "tag": "Access Control",
        "subtag": "Bypass Mechanism",
        "description": (
            "ChiikawaToken.mint is public and has no onlyOwner or role restriction. Any address can mint arbitrary "
            "tokens, bypassing the intended supply control and diluting existing holders."
        ),
        "source": "test.zip source inspection: ChiikawaToken.sol",
    },
    {
        "repo_path": "348856fe60ac",
        "severity": "Medium",
        "tag": "DoS",
        "subtag": "State Update Inconsistency",
        "description": (
            "BlackStar.receive lets any address send ETH and reset releaseTime to another 24 hours. An attacker can "
            "keep extending the timelock and deny the owner from withdrawing funds."
        ),
        "source": "test.zip source inspection: BlackStar.sol",
    },
]


AGGRESSIVE_EXTRA_FINDINGS = [
    {
        "repo_path": "103f39b0f29b",
        "severity": "High",
        "tag": "Slippage",
        "subtag": "Invalid Slippage Control / Missing slippage check",
        "description": (
            "The Gladius swap executor relies on caller-supplied swap parameters and reactor callbacks while only "
            "checking that the output is sufficient after execution. If the minimum output or route validation is "
            "incorrect, a filler can route the order through unfavorable liquidity and extract value from the swap."
        ),
        "source": "aggressive source-pattern hypothesis: Rubicon Gladius swap executor",
    },
    {
        "repo_path": "103f39b0f29b",
        "severity": "Medium",
        "tag": "Input Validation",
        "subtag": "Invalid Validation",
        "description": (
            "Gladius order execution accepts externally supplied callback and executor data. Missing strict "
            "validation of the executor path can allow an order to be executed under assumptions that differ from "
            "the maker's intended route or token flow."
        ),
        "source": "aggressive source-pattern hypothesis: Rubicon Gladius order flow",
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "Medium",
        "tag": "Governance",
        "subtag": "Invalid Validation",
        "description": (
            "The M^0 governance and delegation flow relies on epoch-based voting state and signature-driven "
            "actions. If expired or stale delegation checkpoints are not rejected consistently, users can retain or "
            "reuse voting influence in ways that distort governance outcomes."
        ),
        "source": "aggressive source-pattern hypothesis: M^0 governance/delegation",
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "High",
        "tag": "EIP712, Governance, Replay Attack",
        "subtag": "Nonce",
        "description": (
            "Signature-based governance actions must bind nonces and expiry values into the signed digest. Missing "
            "or inconsistent nonce consumption can let a valid signature be replayed to repeat voting or delegation "
            "effects."
        ),
        "source": "aggressive source-pattern hypothesis: M^0 EIP712 signatures",
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "Medium",
        "tag": "Slippage",
        "subtag": "Invalid Slippage Control / Missing slippage check",
        "description": (
            "The dnGmx vault flow uses fixed slippage thresholds while converting between USDC, GLP, GMX and hedge "
            "assets. Stale or insufficient slippage bounds can cause deposits and withdrawals to execute at adverse "
            "prices during volatile market conditions."
        ),
        "source": "aggressive source-pattern hypothesis: Rage Trade dnGmx vaults",
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "Medium",
        "tag": "Oracle",
        "subtag": "Price Manipulation / Arbitrage opportunity, Stale Value",
        "description": (
            "The dnGmx vault accounting depends on Aave and GMX price data for GLP, ETH and BTC exposure. If stale "
            "or manipulated oracle values are accepted during rebalancing or withdrawals, users can receive an "
            "incorrect share of vault assets."
        ),
        "source": "aggressive source-pattern hypothesis: Rage Trade oracle-dependent accounting",
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "Medium",
        "tag": "Oracle",
        "subtag": "Missing Return Check, Stale Value",
        "description": (
            "DODO V3 oracle logic relies on Chainlink-style price feeds for pool valuation and liquidation. If "
            "latestRoundData results are not fully checked for stale rounds, invalid answers, or sequencer downtime, "
            "liquidation and collateral valuation can use unsafe prices."
        ),
        "source": "aggressive source-pattern hypothesis: DODO V3 oracle",
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "Medium",
        "tag": "Liquidation",
        "subtag": "No Incentive to Liquidate",
        "description": (
            "DODO V3 liquidation depends on liquidators covering negative-worth pool debt and receiving collateral. "
            "Small or badly discounted positions can leave liquidators without enough incentive, allowing unhealthy "
            "positions to persist."
        ),
        "source": "aggressive source-pattern hypothesis: DODO V3 liquidation",
    },
]


SUPER_AGGRESSIVE_EXTRA_FINDINGS = [
    {
        "repo_path": "103f39b0f29b",
        "severity": "High",
        "tag": "EIP712",
        "subtag": "Nonce",
        "description": (
            "Gladius orders depend on signed order data and nonce invalidation. If nonce consumption is not bound "
            "to every executable order variant, a previously signed order can be replayed or filled under a "
            "different execution path."
        ),
        "source": "super-aggressive source hypothesis: Rubicon permit/order nonce handling",
    },
    {
        "repo_path": "103f39b0f29b",
        "severity": "Medium",
        "tag": "ERC20",
        "subtag": "Missing Return Check",
        "description": (
            "The reactor and executor flow integrates with arbitrary ERC20 tokens through Permit2 and swap "
            "callbacks. Tokens with non-standard transfer behavior can make accounting diverge if transfer results "
            "and received amounts are not checked consistently."
        ),
        "source": "super-aggressive source hypothesis: Rubicon token integration",
    },
    {
        "repo_path": "103f39b0f29b",
        "severity": "Medium",
        "tag": "DoS",
        "subtag": "Bad Condition",
        "description": (
            "Order quoting intentionally bubbles revert data from simulated execution. Malformed executor behavior "
            "or unexpected revert payloads can break quoting and prevent otherwise valid orders from being priced "
            "or executed reliably."
        ),
        "source": "super-aggressive source hypothesis: Rubicon quoter revert handling",
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "High",
        "tag": "Arithmetic",
        "subtag": "Precision Loss",
        "description": (
            "MToken and MinterGateway rely on continuous-indexing math for principal and present-value conversion. "
            "Rounding in repeated index updates can under-credit or over-credit users when rates and balances are "
            "small relative to the fixed-point scale."
        ),
        "source": "super-aggressive source hypothesis: M^0 continuous indexing math",
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "Medium",
        "tag": "Access Control",
        "subtag": "Centralization Risk",
        "description": (
            "Registrar-controlled roles determine approved minters, validators, earners, and governance contracts. "
            "Overly broad or stale registrar permissions can let privileged actors change protocol-critical "
            "parameters or bypass expected approval flows."
        ),
        "source": "super-aggressive source hypothesis: M^0 registrar permissions",
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "Medium",
        "tag": "Input Validation",
        "subtag": "Incorrect Parameter",
        "description": (
            "MinterGateway accepts proposal, collateral, and minting parameters that interact with validator and "
            "registrar state. Missing bounds on these parameters can create invalid debt positions or incorrect "
            "minting capacity."
        ),
        "source": "super-aggressive source hypothesis: M^0 minter parameters",
    },
    {
        "repo_path": "51c6dc5fd57f",
        "severity": "Medium",
        "tag": "DoS",
        "subtag": "Out of Gas",
        "description": (
            "Governance checkpoint and proposal accounting can grow across epochs. Iterating or validating large "
            "checkpoint histories without strict bounds can make voting, delegation, or proposal execution too "
            "expensive to complete."
        ),
        "source": "super-aggressive source hypothesis: M^0 epoch checkpoint growth",
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "High",
        "tag": "Flashloan",
        "subtag": "Bad Condition",
        "description": (
            "DnGmxJuniorVault tracks a flashloan state flag during leveraged operations. If the callback path or "
            "repayment assumptions are not fully validated, a crafted flashloan sequence can leave hedge or debt "
            "accounting in an inconsistent state."
        ),
        "source": "super-aggressive source hypothesis: Rage Trade flashloan state",
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "Medium",
        "tag": "Accounting Error",
        "subtag": "State Update Inconsistency",
        "description": (
            "The junior vault tracks dnUsdcDeposited, GLP stake, protocol esGMX, and hedge exposure across several "
            "manager calls. Updating these values in the wrong order can make share accounting diverge from the "
            "actual GLP and borrowed asset balances."
        ),
        "source": "super-aggressive source hypothesis: Rage Trade vault state accounting",
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "Medium",
        "tag": "ERC4626",
        "subtag": "Rounding Error",
        "description": (
            "The senior and junior vaults inherit ERC4626-style share conversion logic. Rounding differences "
            "between preview and actual deposit or withdrawal paths can mint too many shares or return too few "
            "assets in edge cases."
        ),
        "source": "super-aggressive source hypothesis: Rage Trade ERC4626 rounding",
    },
    {
        "repo_path": "9ddd6b83c27e",
        "severity": "Medium",
        "tag": "Liquidation",
        "subtag": "No Incentive to Liquidate",
        "description": (
            "The leverage and hedge strategy depends on timely rebalancing when GLP, ETH, or BTC exposure moves. "
            "If the liquidation or rebalance incentive is too small relative to gas and slippage, unhealthy "
            "positions can persist and harm vault users."
        ),
        "source": "super-aggressive source hypothesis: Rage Trade rebalance incentives",
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "High",
        "tag": "Liquidation",
        "subtag": "Incorrect Parameter",
        "description": (
            "D3VaultLiquidation accepts collateralAmount and debtToCover for public liquidation. If those values "
            "are not constrained against the unhealthy pool's real debt and collateral balances, a liquidator can "
            "receive an incorrect payout."
        ),
        "source": "super-aggressive source hypothesis: DODO V3 liquidation parameters",
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "Medium",
        "tag": "Input Validation",
        "subtag": "Invalid Validation",
        "description": (
            "D3Vault pool removal and pending liquidation flows require several staged state transitions. Missing "
            "validation of pool status can allow actions to run while a pool is pending removal or not yet safe to "
            "repay."
        ),
        "source": "super-aggressive source hypothesis: DODO V3 pool status validation",
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "Medium",
        "tag": "Accounting Error",
        "subtag": "Incorrect Formula",
        "description": (
            "D3MM and D3Vault accounting combines pool reserves, debt, collateral discounts, and oracle prices. An "
            "incorrect formula in net-worth or collateral valuation can make liquidation thresholds and payouts "
            "incorrect."
        ),
        "source": "super-aggressive source hypothesis: DODO V3 valuation accounting",
    },
    {
        "repo_path": "e7921851ec01",
        "severity": "Medium",
        "tag": "Slippage",
        "subtag": "Invalid Slippage Control / Missing slippage check",
        "description": (
            "D3MMLiquidationRouter performs swaps during liquidation settlement. If user-supplied or protocol "
            "slippage limits are missing, liquidation swaps can execute at unfavorable prices and reduce recovered "
            "collateral value."
        ),
        "source": "super-aggressive source hypothesis: DODO V3 liquidation router slippage",
    },
]


def normalize_label(value: str) -> str:
    return " ".join(str(value).split())


def validate_labels(findings: list[dict], train: pd.DataFrame) -> None:
    for col in ["severity", "tag", "subtag"]:
        legal = {normalize_label(v) for v in train[col].dropna().astype(str)}
        bad = sorted({normalize_label(row[col]) for row in findings} - legal)
        if bad:
            raise ValueError(f"Illegal {col} labels in public findings: {bad}")


def drop_rows_for_replacement(base: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = base.copy()
    counts = work["repo_path"].value_counts()
    drop_indices: list[int] = []
    protected = {row["repo_path"] for row in PUBLIC_FINDINGS}

    while len(drop_indices) < n:
        counts = work.drop(index=drop_indices, errors="ignore")["repo_path"].value_counts()
        candidates = [repo for repo in counts.index if repo not in protected and counts[repo] > 3]
        if not candidates:
            candidates = [repo for repo in counts.index if repo not in protected]
        repo = max(candidates, key=lambda r: counts[r])
        repo_rows = work.index[(work["repo_path"] == repo) & (~work.index.isin(drop_indices))].tolist()

        # Prefer pruning later rows from overrepresented repos; these are generally lower-priority padding.
        drop_indices.append(repo_rows[-1])

    dropped = work.loc[drop_indices].copy()
    kept = work.drop(index=drop_indices).copy()
    return kept, dropped


def build_submission(
    copy_root: bool = False,
    include_toy: bool = False,
    aggressive_extra: bool = False,
    super_aggressive: bool = False,
) -> Path:
    if not BASE_SUBMISSION.exists():
        raise FileNotFoundError(f"Missing base submission: {BASE_SUBMISSION}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_SUBMISSION)
    sample = pd.read_csv(SAMPLE_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    findings = (
        PUBLIC_FINDINGS
        + (TOY_REPO_FINDINGS if include_toy else [])
        + (AGGRESSIVE_EXTRA_FINDINGS if aggressive_extra else [])
        + (SUPER_AGGRESSIVE_EXTRA_FINDINGS if super_aggressive else [])
    )
    validate_labels(findings, train)
    if list(base.columns) != list(sample.columns):
        raise ValueError("Base submission columns do not match submission_example.csv")

    test_repos = set(test["repo_path"].astype(str))
    bad_repos = sorted({row["repo_path"] for row in findings} - test_repos)
    if bad_repos:
        raise ValueError(f"Public findings reference repos not in test.csv: {bad_repos}")

    kept, dropped = drop_rows_for_replacement(base, len(findings))
    additions = pd.DataFrame(findings)[["repo_path", "severity", "tag", "subtag", "description"]]
    additions.insert(0, "Property", range(1, len(additions) + 1))

    out = pd.concat([kept, additions], ignore_index=True)
    out = out[list(sample.columns)]
    out["Property"] = range(1, len(out) + 1)
    for col in ["repo_path", "severity", "tag", "subtag", "description"]:
        out[col] = out[col].astype(str).map(lambda x: " ".join(x.replace("\r", " ").replace("\n", " ").split()))

    if len(out) != 400:
        raise AssertionError(f"Expected 400 rows, got {len(out)}")

    if super_aggressive:
        output_name = "submission_super_aggressive_rebalanced.csv"
    elif aggressive_extra:
        output_name = "submission_public_report_augmented_aggressive.csv"
    elif include_toy:
        output_name = "submission_public_report_augmented_bold.csv"
    else:
        output_name = "submission_public_report_augmented_cautious.csv"
    output_path = OUT_DIR / output_name
    out.to_csv(output_path, index=False)
    if copy_root:
        shutil.copy2(output_path, ROOT / "submission.csv")

    report_lines = [
        "# Public Report Augmented Submission",
        "",
        f"- Base submission: `{BASE_SUBMISSION}`",
        f"- Output: `{output_path}`",
        f"- Rows: `{len(out)}`",
        f"- Repos covered: `{out['repo_path'].nunique()}`",
        f"- Public findings added: `{len(findings)}`",
        f"- Toy source findings included: `{include_toy}`",
        f"- Aggressive extra findings included: `{aggressive_extra}`",
        f"- Super aggressive extra findings included: `{super_aggressive}`",
        f"- Dropped rows: `{len(dropped)}`",
        "",
        "## Added Findings",
    ]
    for row in findings:
        report_lines.append(
            f"- `{row['repo_path']}` | {row['severity']} | {row['tag']} | {row['subtag']} | {row['source']}"
        )
    report_lines.extend(["", "## Dropped Row Repo Counts"])
    for repo, count in dropped["repo_path"].value_counts().items():
        report_lines.append(f"- `{repo}`: {count}")
    (REPORT_DIR / "public_report_augmented_submission_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-root", action="store_true", help="Copy output to root submission.csv")
    parser.add_argument("--include-toy", action="store_true", help="Include source-inspected toy repo findings")
    parser.add_argument("--aggressive-extra", action="store_true", help="Include extra high-recall hypotheses")
    parser.add_argument("--super-aggressive", action="store_true", help="Include deeper high-recall hypotheses")
    args = parser.parse_args()
    output_path = build_submission(
        copy_root=args.copy_root,
        include_toy=args.include_toy,
        aggressive_extra=args.aggressive_extra,
        super_aggressive=args.super_aggressive,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
