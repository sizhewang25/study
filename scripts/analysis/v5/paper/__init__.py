"""The paper's numbers, read from the v5 artifacts and written the way the paper writes them.

`report-paper --group <id> --section N` prints every statistic one paper section
quotes, in the section's order, each formatted by one set of rules (`fmt`) from
the unrounded artifact value, with the file it came from. It does no analysis
-- a statistic with no artifact column gets one in its module first -- and it
needs nothing from the paper's sources: comparing the report with the text is
left to its reader.

One module per section (`s4_error_distance`, ...); `core` holds the claim
model, the artifact paths under the group's pooled names and the Markdown/JSON
writer.
"""
