# SPEC task48c — board nits seen on casino-878 after task48b
1. Decisive card body still carries the dashboard wording ("1수 B 악수(??) — 거절, 사정 호소 뒤 관계 점수 50 → 8"). Use plain
   language only, same `#n` numbering as the header, no glyphs: "거절하며 사정을 호소했지만, B의 관계 점수가 50 → 8로 떨어졌다".
   For a well-received turn: "…해서 A의 관계 점수가 45 → 61로 올랐다".
2. Tide flags show 1·2·3 while cards show #1·#3·#4. Label flags with the turn number (#1, #3, #4) so they match the cards.
3. Lesson block: keep the sentence, add the source in small type after it ("— Fisher & Ury, Getting to Yes") since it is the
   only theory line on the page; everything else stays citation-free.
tests: card body has no "수 " ordinal and no "(??)"; flag labels equal card turn numbers; lesson has a source span.
