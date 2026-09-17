"""Independently worded pilot prompts. These are candidates, not verified gold."""
from pathlib import Path
from . import corpus

FACTUAL = [
    ("1.5", "Which risks must a national bank consider before undertaking a securities investment?"),
    ("1.5", "What evidence about a securities obligor and what examination records must a national bank maintain?"),
    ("1.6", "May a national bank buy a security that the issuer can choose to convert into stock?"),
    ("2.3", "What restrictions govern personal receipt of credit life insurance income by bank insiders involved in selling the insurance?"),
    ("2.4", "How is the annual cap on a bank employee's credit life insurance sales incentive calculated?"),
    ("2.5", "How must a licensed bank insider compensate the bank when selling credit life insurance to the bank's loan customers?"),
    ("9.4", "Who directs a national bank's fiduciary activities, and may fiduciary functions be delegated?"),
    ("9.4", "Under what conditions can a national bank use outside personnel or services for fiduciary activities?"),
    ("9.5", "What subjects should a national bank address in written fiduciary policies and procedures?"),
    ("9.5", "What is the purpose of a national bank's written policies governing fiduciary activities?"),
    ("9.6", "What review must occur before a national bank accepts a fiduciary account?"),
    ("9.6", "What initial and periodic asset reviews apply when a national bank has fiduciary investment discretion?"),
    ("9.8", "How long must a national bank retain fiduciary account records after account termination or related litigation?"),
    ("9.8", "How should fiduciary account records be kept separate from the bank's other records?"),
    ("9.9", "How often must significant fiduciary activities be audited, and what alternative audit arrangement is allowed?"),
    ("9.9", "What independence requirements apply to the fiduciary audit committee?"),
    ("9.10", "How long may fiduciary funds awaiting investment or distribution remain idle?"),
    ("9.10", "What safeguards apply to fiduciary funds deposited with the bank itself while awaiting investment or distribution?"),
    ("9.13", "How many designated fiduciary officers or employees must share custody or control of fiduciary assets?"),
    ("9.13", "When may a national bank maintain fiduciary investments off its premises?"),
    ("12.3", "What is the minimum retention period for customer securities transaction records?"),
    ("12.3", "What information belongs in chronological records of customer securities purchases and sales?"),
    ("12.4", "When must a national bank send a securities transaction notification if it uses a registered broker's confirmation?"),
    ("12.4", "What information must a national bank's own securities transaction confirmation contain?"),
    ("14.30", "What insurance-related representations about obtaining bank credit are prohibited?"),
    ("14.30", "What misleading representations about insurance, annuities, and federal backing must a covered seller avoid?"),
    ("14.40", "What disclosures must accompany the initial purchase of an insurance product or annuity from a covered bank seller?"),
    ("14.40", "What acknowledgment must a covered insurance seller obtain from the consumer concerning required disclosures?"),
    ("21.21", "What core components must an OCC-supervised bank's BSA compliance program include?"),
    ("21.21", "What board approval and written-program requirements apply to a bank's BSA compliance procedures?"),
    ("22.3", "What minimum amount of flood insurance is generally required for a designated loan?"),
    ("22.3", "What loan actions trigger the requirement for flood insurance where coverage is available?"),
    ("22.5", "When must a lender escrow flood insurance premiums and fees for a residential designated loan?"),
    ("22.5", "What exceptions should a lender consider before imposing flood insurance escrow?"),
    ("22.7", "What notice and waiting period precede force placement of flood insurance?"),
    ("22.7", "What must the lender do after receiving evidence that a borrower has obtained flood insurance coverage?"),
    ("22.9", "Who must receive notice when collateral for a loan is in a special flood hazard area?"),
    ("22.9", "What must a special flood hazard notice communicate about insurance and disaster relief?"),
    ("32.3", "What is the general single-borrower lending limit and the additional allowance for qualifying secured lending?"),
    ("32.5", "When are loans to separate persons combined for lending-limit purposes?"),
]

APPLICATION = [
    ("1.5", "An investment desk has attractive price quotes but no evidence of the issuer's ability to pay. Explain the due diligence and records it needs before proceeding."),
    ("1.6", "A bond gives the issuing company the right to replace repayment with its own shares. Can a national bank purchase it? Explain the relevant restriction."),
    ("2.3", "A loan officer personally receives income for selling credit life insurance to the bank's borrowers. Identify the restrictions that compliance should investigate."),
    ("2.4", "A bank officer earns an annual salary of $80,000; participating loan officers average $100,000. Calculate the maximum annual credit life insurance sales incentive."),
    ("2.4", "A bank proposes paying a credit life insurance sales bonus equal to eight percent of an employee's salary. What additional salary information and limit must it check?"),
    ("2.5", "A licensed bank director sells credit life insurance to bank borrowers and proposes keeping the income while paying a small premises fee. Assess that arrangement."),
    ("9.4", "A board delegates fiduciary operations to a committee. Explain the governance responsibilities and delegation authority that apply."),
    ("9.5", "A trust department relies on informal practices rather than written procedures. Identify the compliance gap and the subjects a written framework should address."),
    ("9.6", "A bank accepts a complex trust without first checking whether it can administer the assets. Identify the missed review and what should have occurred."),
    ("9.6", "A newly accepted discretionary fiduciary account contains concentrated investments. Explain the initial asset review and subsequent review cycle."),
    ("9.8", "A fiduciary account closed four years ago, but related litigation ended last year. May the bank destroy the account records now? Explain the retention calculation."),
    ("9.8", "A bank mixes trust account records with its general business files without separate identification. What recordkeeping arrangement is required?"),
    ("9.9", "A bank wants to replace an annual fiduciary audit with ongoing audits. Explain the conditions for an acceptable continuous audit system."),
    ("9.9", "The proposed fiduciary audit committee includes an officer who directly manages the audited activities. Identify the independence issue to check."),
    ("9.10", "Cash in an account over which the bank has investment discretion has remained idle for months. What standard governs the delay?"),
    ("9.10", "A fiduciary department places client cash in its own bank pending investment. Explain the applicable safeguards and exceptions."),
    ("9.13", "One employee has sole control of a trust account's securities. Is that custody arrangement sufficient? Explain any relevant exception."),
    ("9.13", "A bank wants to hold fiduciary investments at an off-premises location. What conditions must its custody controls meet?"),
    ("12.3", "A national bank plans to delete customer securities trade records after two years. Assess the proposed retention schedule."),
    ("12.4", "A bank receives a broker's customer trade confirmation on Monday. Explain the bank's deadline for forwarding it, assuming Tuesday is a business day."),
    ("14.30", "A lender tells an applicant that approval depends on buying insurance from its affiliate. Identify the applicable restriction and avoid assuming that all insurance requirements are prohibited."),
    ("14.40", "A customer assumes an annuity bought at a bank is an insured bank deposit. What disclosures should the covered seller provide?"),
    ("21.21", "A bank has transaction-monitoring software but no designated compliance officer or training program. Identify the missing BSA program controls."),
    ("21.21", "Management adopts a BSA program but never submits it for board approval. What governance step is missing?"),
    ("22.3", "A designated loan has a $150,000 balance and the applicable available coverage limit is $250,000. Explain the minimum-coverage calculation and any relevant property-value constraint."),
    ("22.5", "A lender originates a residential designated loan and proposes collecting flood premiums directly instead of escrowing. What requirements and exceptions must it assess?"),
    ("22.7", "During a loan's term, the servicer discovers inadequate flood coverage. Describe the notice and purchase steps it must follow."),
    ("22.7", "A borrower supplies proof of overlapping coverage after the lender force-places flood insurance. Explain the cancellation and refund duties."),
    ("22.9", "A bank finances a building in a special flood hazard area where federal flood insurance is unavailable. Does that remove the written-notice obligation?"),
    ("32.5", "A loan is nominally made to one company, but its proceeds directly benefit another. Explain how that affects lending-limit attribution."),
]

POLICY = [
    ("1.5", "Draft a short securities-investment due-diligence and examination-records policy for a national bank."),
    ("1.6", "Write an investment-policy clause controlling purchases of issuer-convertible securities."),
    ("2.3", "Draft a policy governing insiders' receipt of credit life insurance sales income."),
    ("2.4", "Draft a credit life insurance incentive-plan control covering calculation and approval of annual employee payouts."),
    ("2.5", "Write a policy clause governing compensation to the bank from licensed insiders selling credit life insurance to borrowers."),
    ("9.4", "Draft a fiduciary governance policy addressing board direction and delegation of functions."),
    ("9.5", "Outline the required written policy framework for a national bank exercising fiduciary powers."),
    ("9.6", "Draft a fiduciary account review procedure covering acceptance, initial asset review, and recurring review."),
    ("9.8", "Write a fiduciary records policy covering documentation, retention, and separation of records."),
    ("9.9", "Draft a fiduciary audit policy covering audit frequency, reporting, and committee independence."),
    ("9.10", "Draft a policy for fiduciary cash awaiting investment or distribution, including placement with the bank itself."),
    ("9.13", "Write a custody policy for fiduciary assets covering joint control and off-premises holdings."),
    ("12.3", "Draft a customer securities transaction recordkeeping policy with retention and required record categories."),
    ("12.4", "Draft a securities trade notification procedure addressing timing and required confirmation content."),
    ("14.30", "Write a consumer insurance sales policy addressing coercion, tying, and misleading representations."),
    ("14.40", "Draft an insurance and annuity disclosure procedure covering delivery and consumer acknowledgment."),
    ("21.21", "Draft a concise BSA compliance-program governance policy for an OCC-supervised bank."),
    ("22.5", "Draft a flood insurance escrow procedure that requires staff to assess applicability and exceptions."),
    ("22.7", "Draft a flood insurance force-placement procedure covering notice, purchase, cancellation, and refunds."),
    ("22.9", "Write a special flood hazard notice procedure identifying recipients, required content, timing, and records."),
]

OUTSIDE = [
    "Draft a complete Truth in Lending annual percentage rate disclosure policy for consumer mortgages.",
    "Explain the federal electronic fund transfer error-resolution deadlines for consumer checking accounts.",
    "Describe the mortgage servicing notice requirements under Regulation X.",
    "Explain the Equal Credit Opportunity Act adverse-action notice rules in Regulation B.",
    "List the federal Home Mortgage Disclosure Act reporting data fields for a loan application.",
    "Explain the Federal Reserve's Regulation D rules for transaction accounts.",
    "Describe the FDIC deposit insurance coverage limits for joint ownership accounts.",
    "Provide the NCUA rules governing federal credit union membership eligibility.",
    "Explain the Internal Revenue Service rules for deducting residential mortgage interest.",
    "Draft a state-law foreclosure notice for a residential property in California.",
]


def draft(cfg):
    if cfg["ecfr"].get("chapter") != "I":
        raise ValueError("These pilot prompts target Chapter I; author a scope-appropriate benchmark for all-title runs")
    directory = Path(cfg["experiment_dir"])
    docs = {d["number"]: d for d in corpus.read_rows(directory / "documents.jsonl") if d["kind"] == "SECTION"}
    rows = []
    for kind, prompts in [("factual", FACTUAL), ("application", APPLICATION), ("policy", POLICY)]:
        for number, question in prompts:
            d = docs[number]
            rows.append({"id": f"question-{len(rows)+1:03}", "type": kind, "question": question,
                         "reference_answer": "", "required_claims": [], "acceptable_citations": [d["citation"]], "granularity": "section",
                         "evidence": [{"document_id": d["id"], "start": 0, "end": len(d["text"]), "quote": d["text"]}],
                         "review_status": "pending", "reviewer": ""})
    for question in OUTSIDE:
        rows.append({"id": f"question-{len(rows)+1:03}", "type": "outside_scope", "question": question,
                     "reference_answer": "The requested governing rules are outside the Chapter I corpus; identify that limitation rather than inventing an OCC citation.",
                     "required_claims": ["Identify that the requested governing rules extend beyond the Chapter I corpus."],
                     "acceptable_citations": [], "granularity": "none", "evidence": [],
                     "scope_reason": "Primary requested rules are outside Title 12 Chapter I; incidental cross-references do not supply the complete requested rule.",
                     "review_status": "pending", "reviewer": ""})
    path = directory / "benchmark_candidates.jsonl"
    if path.exists():
        raise ValueError("Candidate file already exists; do not overwrite review work")
    corpus.write_rows(path, rows)
    print(f"Wrote {len(rows)} candidate questions with source passages; answer keys and verification are still required")
