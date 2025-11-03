# Writing Guidelines for Technical Manuscript

This document provides guidelines for maintaining a neutral, academic tone free from LLM-sounding constructions.

## 1. Avoid Meta-Announcement Phrases

### DON'T Use:
- "The key insight is that..."
- "The key idea is that..."
- "The key point is that..."
- "The key takeaway is..."
- "The main takeaway is..."
- "Another key aspect/feature/point..."
- "An important point here is..."
- "The key is to/that..."

### DO Use:
- State the fact directly
- "It turns out that..."
- "What this tells us is..."
- "What matters here is that..."
- "We need to..."
- Just begin with the statement

### Examples:
- ❌ "The key insight is that the Bellman operator is a contraction."
- ✅ "The Bellman operator is a contraction."

- ❌ "The key is to choose elements that work well together."
- ✅ "We should choose elements that work well together."

## 2. Avoid Hedging Phrases

### DON'T Use:
- "it is worth noting that..."
- "it is important to note that..."
- "it should be noted that..."

### DO Use:
- State directly without the hedge
- In rare cases where context is needed: "Note that..." or "In our setting,..."

### Examples:
- ❌ "It is worth noting that these concepts extend to stochastic problems."
- ✅ "These concepts extend to stochastic problems."

## 3. Avoid Overly Dramatic Transitions

### DON'T Use:
- "Remarkably,..."
- "Interestingly,..."
- "Crucially,..."
- "Fascinatingly,..."
- When they introduce nothing actually remarkable

### DO Use:
- Just state the fact
- In rare cases where genuinely surprising: acceptable, but use sparingly

### Examples:
- ❌ "Interestingly, the first-order conditions turn out to be equivalent..."
- ✅ "The first-order conditions turn out to be equivalent..."

## 4. Avoid Meta-Demonstration Verbs

### DON'T Overuse:
- "reveals that..."
- "clarifies that..."
- "demonstrates that..."
- "illustrates that..."
- "highlights that..."
- "emphasizes that..."
- "underscores that..."

### DO Use:
- "shows that..."
- "tells us..."
- "makes clear..."
- "indicates..."
- Or just state the fact directly

### Examples:
- ❌ "The connection reveals that LSTD is Galerkin projection..."
- ✅ "This shows that LSTD is Galerkin projection..."

- ❌ "This bound reveals that the approximation error depends on..."
- ✅ "This bound shows that the approximation error depends on..."

## 5. Avoid Flowery Adjectives

### DON'T Overuse:
- "elegant solution/variant/approach"
- "beautiful connection/result"
- "powerful method/tool/result"
- "remarkable property/result"
- "fascinating question/observation"
- "stunning result"
- "impressive performance"
- "wonderful property"
- "sophisticated method" (when "complex" is more accurate)

### DO Use:
- "useful", "effective", "efficient"
- "clean", "simple", "direct"
- "complex", "involved"
- Or just describe the property without adjectives

### Examples:
- ❌ "A particularly elegant variant is orthogonal collocation, which exploits a beautiful connection..."
- ✅ "Orthogonal collocation exploits a useful connection..."

- ❌ "This remarkable result explains why..."
- ✅ "This explains why..."

## 6. Avoid Escalating "Not Only...But" Constructions

### DON'T Use:
- "not only X but also Y"
- "not just X but Y"
- "not merely X but Y"
- "beyond just X"
When trying to sound impressive

### DO Use:
- "both X and Y"
- "X and Y"
- State directly without the escalation

### Examples:
- ❌ "The goal is not just to solve it once, but to understand how..."
- ✅ "The goal is to understand how..."

- ❌ "not only in terms of state evolution but also in terms of cost"
- ✅ "in both state evolution and cost"

## 7. Replace Em Dashes Used for Pacing

### DON'T Use:
- Em dashes (—) for dramatic pacing or emphasis in prose

### DO Use:
- A period and new sentence
- A comma if the clause is short
- Keep technical dashes in math/code

### Examples:
- ❌ "This is convenient — it lets us reuse the same operator."
- ✅ "This is convenient. It lets us reuse the same operator."
- ✅ "This is convenient, because it lets us reuse the same operator."

## 8. Avoid Excessive Bold Formatting

### DON'T Bold:
- Complete statement sentences for emphasis
- Rhetorical questions ("**Why is this useful?**")
- Mid-sentence phrases just for emphasis
- Declarative statements that announce importance

### DO Bold:
- Technical terms being defined
- Method names in lists
- Algorithm step labels (Step 1, Step 2)
- Table headers
- True emphasis (sparingly)

### Examples:
- ❌ "**The same principle extends to functions.** A function R..."
- ✅ "The same principle extends to functions. A function R..."

- ❌ "**Why is this useful?** It transforms..."
- ✅ "Why is this useful? It transforms..."

## 9. Simplify Clichéd Section Titles

### DON'T Use:
- "Method X: Best of Both Worlds"
- "Approach Y: An Alternative Framework"
- "Beyond the X: Y and Extensions"
- "The X Condition: Y Approximators"
- Other "Title: Catchy Subtitle" patterns that sound like blog posts

### DO Use:
- Simple, descriptive titles
- "Method X" or just "X"
- "Weighted Norms and Extensions" (not "Beyond the Sup Norm: ...")

### Exceptions (OK to use):
- Numbered steps: "Step 1: Choose a Basis"
- Examples: "Example: Optimal Harvest"
- Technical classifications: "Perspective 1: Max of Affine Maps"
- Method descriptions: "Galerkin Method: Test Against the Basis" (informative)

## 10. Avoid "This is precisely"

### DON'T Use:
- "This is precisely the X..."
- "This is precisely why..."

### DO Use:
- "We have just derived the X..."
- "This gives us the X..."
- "We recover the X..."
- "This matches the X..."

### Examples:
- ❌ "This is precisely the LSTD solution."
- ✅ "We have just derived the LSTD solution."

## 11. Avoid Filler Words

### DON'T Use:
- "clearly"
- "obviously"  
- "of course"
Unless genuinely necessary for pedagogical clarity

### DO Use:
- State the fact without the filler
- If needed: "Note that...", "We see that..."

## 12. Use Proper ASCII Characters

### DON'T Use:
- Unicode smart quotes: ' ' " "
- Em dashes for pacing: —

### DO Use:
- ASCII apostrophe: '
- ASCII quotes: "
- Periods, commas, or semicolons for sentence structure
- Keep em dashes for technical use (ranges, etc.)

## General Principles

1. **Be direct**: State facts without announcing their importance
2. **Be neutral**: Avoid hype and enthusiasm in technical exposition
3. **Be precise**: Use technical language appropriately
4. **Be pedagogical**: Guide the reader without being chatty
5. **Let content speak**: Don't tell readers what's important—show them through clear exposition

## Tone Guidelines

### Target Tone:
- Neutral, academic
- Explanatory without being condescending
- Technical without being obscure
- Accessible without being casual

### Avoid:
- Marketing language ("powerful", "revolutionary", "game-changing")
- Chatty constructions ("Here's the thing...", "Now, here's where it gets interesting...")
- Excessive signposting ("As we will see...", "It turns out that...", "What this means is...")
- Artificial enthusiasm ("This is a remarkable idea!", "Fascinatingly,...")

## When in Doubt

Ask: "Would this sentence appear in a 1990s textbook?" If the answer is no because it sounds too much like a blog post, presentation slide, or marketing copy, revise it.

Good academic writing is clear, direct, and lets the mathematics and ideas carry the weight. It doesn't need to constantly announce what's important or sound excited about every result.

