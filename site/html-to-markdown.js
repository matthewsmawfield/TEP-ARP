#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * HTML to Markdown Converter for TEP-ARP Site
 * Converts the built static HTML site into a clean markdown document
 */

class HTMLToMarkdownConverter {
    constructor() {
        this.output = '';
        this.currentSection = '';
    }

    /**
     * Convert HTML string to markdown with proper academic formatting
     */
    htmlToMarkdown(html) {
        // Remove script tags and their content
        html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
        
        // Remove style tags and their content
        html = html.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');
        
        // Remove comments
        html = html.replace(/<!--[\s\S]*?-->/g, '');
        
        // Preserve MathJax expressions before processing
        const mathExpressions = [];
        html = html.replace(/<span[^>]*class=["'][^"']*MathJax[^"']*["'][^>]*>.*?<\/span>/gi, (match) => {
            mathExpressions.push(match);
            return `__MATH_EXPRESSION_${mathExpressions.length - 1}__`;
        });
        
        // Convert manuscript sections to proper markdown structure FIRST
        // build.js uses <section> tags, not <div> tags
        html = html.replace(/<(?:div|section)[^>]*class=["'][^"']*manuscript-section[^"']*["'][^>]*data-section=["']([^"']*)["'][^>]*>/gi, '\n\n## $1\n\n');

        // Convert headers
        html = html.replace(/<h1[^>]*>(.*?)<\/h1>/gi, '\n# $1\n\n');
        html = html.replace(/<h2[^>]*>(.*?)<\/h2>/gi, '\n## $1\n\n');
        html = html.replace(/<h3[^>]*>(.*?)<\/h3>/gi, '\n### $1\n\n');
        html = html.replace(/<h4[^>]*>(.*?)<\/h4>/gi, '\n#### $1\n\n');

        // Protect pre/code blocks BEFORE paragraph conversion (which collapses whitespace)
        const codeBlocks = [];
        html = html.replace(/<pre[^>]*><code[^>]*>([\s\S]*?)<\/code><\/pre>/gi, (match, content) => {
            codeBlocks.push(content);
            return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
        });

        // Convert paragraphs (collapse internal whitespace to avoid markdown code-block indentation)
        html = html.replace(/<p[^>]*>(.*?)<\/p>/gis, (match, inner) => {
            const cleaned = inner.replace(/\s+/g, ' ').trim();
            return `${cleaned}\n\n`;
        });

        // Convert strong/bold (avoid mid-sentence bold in generated markdown)
        html = html.replace(/<(strong|b)[^>]*>(.*?)<\/(strong|b)>/gis, '$2');

        // Convert images
        html = html.replace(/<img[^>]*src=["']([^"']*)["'][^>]*alt=["']([^"']*)["'][^>]*>/gi, (match, src, alt) => {
            // Fix path for root-level markdown: canonical figures
            // live in results/figures/ (dist/figures/ is a copy).
            if (src.startsWith('figures/')) {
                src = 'results/' + src;
            }
            return `\n![${alt}](${src})\n`;
        });

        // Convert emphasis/italic
        html = html.replace(/<(em|i)[^>]*>(.*?)<\/(em|i)>/gi, '*$2*');

        // Convert links
        html = html.replace(/<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)<\/a>/gi, '[$2]($1)');

        // Convert lists
        html = html.replace(/<ul[^>]*>/gi, '\n');
        html = html.replace(/<\/ul>/gi, '\n');
        html = html.replace(/<ol[^>]*>/gi, '\n');
        html = html.replace(/<\/ol>/gi, '\n');
        html = html.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, (match, inner) => {
            const cleaned = inner.replace(/\s+/g, ' ').trim();
            return `- ${cleaned}\n`;
        });

        // Convert blockquotes
        html = html.replace(/<blockquote[^>]*>(.*?)<\/blockquote>/gi, '\n> $1\n\n');

        // Restore code blocks as fenced markdown code blocks
        codeBlocks.forEach((content, index) => {
            html = html.replace(`__CODE_BLOCK_${index}__`, '\n```\n' + content + '\n```\n\n');
        });

        // Convert inline code
        html = html.replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, '`$1`');
        
        // Convert line breaks
        html = html.replace(/<br\s*\/?>/gi, '\n');
        
        // Convert horizontal rules
        html = html.replace(/<hr\s*\/?>/gi, '\n---\n\n');
        
        // Convert divs with special classes to markdown equivalents
        html = html.replace(/<div[^>]*class=["'][^"']*abstract[^"']*["'][^>]*>/gi, '');
        html = html.replace(/<div[^>]*class=["'][^"']*theorem[^"']*["'][^>]*>/gi, '\n**Theorem:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*principle[^"']*["'][^>]*>/gi, '\n');
        html = html.replace(/<div[^>]*class=["'][^"']*proof[^"']*["'][^>]*>/gi, '\n*Proof:*\n');
        html = html.replace(/<div[^>]*class=["'][^"']*experimental-section[^"']*["'][^>]*>/gi, '\n**Experimental Section:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*critical-analysis[^"']*["'][^>]*>/gi, '\n**Critical Analysis:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*significance[^"']*["'][^>]*>/gi, '\n**Significance:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*key-finding[^"']*["'][^>]*>/gi, '\n**Key Finding:**\n');
        html = html.replace(/<div[^>]*class=["'][^"']*validation-box[^"']*["'][^>]*>/gi, '\n**Validation:**\n');
        
        // Handle abstract section specially
        html = html.replace(/## Abstract\s*\n\s*<h2>Abstract<\/h2>/gi, '## Abstract\n\n');
        
        // Convert tables
        html = html.replace(/<table[^>]*>(.*?)<\/table>/gis, (match) => {
            return this.convertTable(match);
        });
        
        html = html.replace(/<(?!\/?[a-zA-Z!])/g, '&lt;');

        // Preserve sub/sup tags before generic stripping
        const subTags = [];
        html = html.replace(/<sub>(.*?)<\/sub>/gi, (match, inner) => {
            subTags.push(inner);
            return `__SUB_${subTags.length - 1}__`;
        });
        const supTags = [];
        html = html.replace(/<sup>(.*?)<\/sup>/gi, (match, inner) => {
            supTags.push(inner);
            return `__SUP_${supTags.length - 1}__`;
        });

        // Remove remaining HTML tags
        html = html.replace(/<[^>]+>/g, '');
        
        // Restore MathJax expressions
        mathExpressions.forEach((expr, index) => {
            html = html.replace(`__MATH_EXPRESSION_${index}__`, expr);
        });
        
        // Restore sub/sup tags
        subTags.forEach((inner, index) => {
            html = html.replace(`__SUB_${index}__`, `<sub>${inner}</sub>`);
        });
        supTags.forEach((inner, index) => {
            html = html.replace(`__SUP_${index}__`, `<sup>${inner}</sup>`);
        });
        
        // Decode HTML entities
        html = html.replace(/&amp;/g, '&');
        // Decode &lt; and &gt; to raw angle brackets for markdown output
        // (in markdown, < and > in math expressions like $z < 0.15$ are safe)
        html = html.replace(/&lt;/g, '<');
        html = html.replace(/&gt;/g, '>');
        html = html.replace(/&quot;/g, '"');
        html = html.replace(/&#39;/g, "'");
        html = html.replace(/&nbsp;/g, ' ');
        html = html.replace(/&mdash;/g, '—');
        html = html.replace(/&ndash;/g, '–');
        html = html.replace(/&hellip;/g, '…');
        html = html.replace(/&oacute;/g, 'ó');
        html = html.replace(/&eacute;/g, 'é');
        html = html.replace(/&aacute;/g, 'á');
        html = html.replace(/&iacute;/g, 'í');
        html = html.replace(/&uacute;/g, 'ú');
        html = html.replace(/&ntilde;/g, 'ñ');
        html = html.replace(/&uuml;/g, 'ü');
        html = html.replace(/&times;/g, '×');
        html = html.replace(/&minus;/g, '−');
        html = html.replace(/&plusmn;/g, '±');
        html = html.replace(/&sup2;/g, '²');
        html = html.replace(/&sup3;/g, '³');
        html = html.replace(/&sup1;/g, '¹');
        html = html.replace(/&deg;/g, '°');
        html = html.replace(/&lambda;/g, 'λ');
        html = html.replace(/&mu;/g, 'μ');
        html = html.replace(/&sigma;/g, 'σ');
        // Decode numeric character references (decimal and hex)
        // e.g. &#8320; → subscript 0, &#8722; → minus sign
        html = html.replace(/&#(\d+);/g, (match, dec) => {
            return String.fromCharCode(parseInt(dec, 10));
        });
        html = html.replace(/&#x([0-9a-fA-F]+);/g, (match, hex) => {
            return String.fromCharCode(parseInt(hex, 16));
        });

        // Remove leading indentation on every line (prevents accidental markdown code blocks)
        html = html.replace(/^[ \t]+/gm, '');
        
        // Clean up whitespace
        html = html.replace(/\n\s*\n\s*\n/g, '\n\n');
        html = html.replace(/^\s+|\s+$/g, '');
        
        // Remove duplicate headers (same header appearing consecutively)
        html = html.replace(/(##\s+[^\n]+)\n+\1/g, '$1');
        
        // Clean up any remaining formatting issues
        html = html.replace(/\n{3,}/g, '\n\n');
        
        return html;
    }

    /**
     * Convert HTML table to markdown table
     */
    convertTable(tableHtml) {
        const rows = [];
        const rowMatches = tableHtml.match(/<tr[^>]*>(.*?)<\/tr>/gis);

        if (!rowMatches) return '';

        rowMatches.forEach((row, index) => {
            const cells = row.match(/<t[dh][^>]*>(.*?)<\/t[dh]>/gis);
            if (cells) {
                const cellTexts = cells.map(cell => {
                    // Escape stray '<' that are not real tags (e.g. math like
                    // $w < 0.026$) before stripping markup, so they are not
                    // mistaken for tag opens and eaten with the next '>'.
                    let text = cell.replace(/<(?!\/?[a-zA-Z!])/g, '&lt;');
                    text = text.replace(/<[^>]+>/g, '');
                    // Decode HTML entities in cell text
                    text = text.replace(/&amp;/g, '&');
                    text = text.replace(/&lt;/g, '<');
                    text = text.replace(/&gt;/g, '>');
                    text = text.replace(/&quot;/g, '"');
                    text = text.replace(/&#39;/g, "'");
                    text = text.replace(/&nbsp;/g, ' ');
                    text = text.replace(/&mdash;/g, '—');
                    text = text.replace(/&ndash;/g, '–');
                    text = text.replace(/&oacute;/g, 'ó');
                    text = text.replace(/&eacute;/g, 'é');
                    text = text.replace(/&times;/g, '×');
                    text = text.replace(/&#(\d+);/g, (m, d) => String.fromCharCode(parseInt(d, 10)));
                    text = text.replace(/&#x([0-9a-fA-F]+);/g, (m, h) => String.fromCharCode(parseInt(h, 16)));
                    return text.trim();
                });
                rows.push(cellTexts);
            }
        });

        if (rows.length === 0) return '';

        // Extract <caption> if present
        let caption = '';
        const captionMatch = tableHtml.match(/<caption[^>]*>([\s\S]*?)<\/caption>/i);
        if (captionMatch) {
            caption = captionMatch[1]
                .replace(/<(?!\/?[a-zA-Z!])/g, '&lt;')
                .replace(/<[^>]+>/g, '')
                .replace(/&amp;/g, '&')
                .replace(/&lt;/g, '<')
                .replace(/&gt;/g, '>')
                .replace(/&quot;/g, '"')
                .replace(/&#39;/g, "'")
                .replace(/&nbsp;/g, ' ')
                .replace(/&mdash;/g, '—')
                .replace(/&ndash;/g, '–')
                .replace(/&oacute;/g, 'ó')
                .replace(/&eacute;/g, 'é')
                .replace(/&times;/g, '×')
                .replace(/&#(\d+);/g, (m, d) => String.fromCharCode(parseInt(d, 10)))
                .replace(/&#x([0-9a-fA-F]+);/g, (m, h) => String.fromCharCode(parseInt(h, 16)))
                .replace(/\s+/g, ' ')
                .trim();
        }

        // Create markdown table (simple format like Jakarta)
        let markdown = '\n';
        if (caption) {
            markdown += `*${caption}*\n\n`;
        }
        rows.forEach((row, index) => {
            markdown += '| ' + row.join(' | ') + ' |\n';
            if (index === 0) {
                // Add separator row
                markdown += '|' + row.map(() => '---').join('|') + '|\n';
            }
        });
        markdown += '\n';

        return markdown;
    }

    /**
     * Extract title and metadata from HTML
     */
    extractMetadata(html) {
        const titleMatch = html.match(/<title[^>]*>(.*?)<\/title>/i);
        const title = titleMatch ? titleMatch[1] : 'Cosmological Voids vs Temporal Shear: An Empirical Falsification of Kinematic Hubble Tension Solutions';
        
        const authorMatch = html.match(/<meta[^>]*name=["']author["'][^>]*content=["']([^"']*)["']/i);
        const author = authorMatch ? authorMatch[1] : 'Matthew Lukin Smawfield';
        
        const versionMatch = html.match(/<div[^>]*class=["'][^"']*version[^"']*["'][^>]*>(.*?)<\/div>/i);
        const version = versionMatch ? versionMatch[1]
            .replace(/<[^>]+>/g, '')
            .replace(/^Version:\s*/i, '')
            .trim() : 'v0.2 (Valencia)';
        
        const dateMatch = html.match(/<div[^>]*class=["'][^"']*date[^"']*["'][^>]*>(.*?)<\/div>/i);
        const date = dateMatch ? dateMatch[1].replace(/<[^>]+>/g, '').trim() : 'First published: 29 August 2026 · Last updated: 30 August 2026';
        
        const doiMatch = html.match(/DOI:\s*<a[^>]*href=["']([^"']*)["'][^>]*>\s*([^<]*?)\s*<\/a>/i);
        const doi = doiMatch ? doiMatch[2] : 'pending assignment';
        
        return { title, author, version, date, doi };
    }

    /**
     * Extract main content from HTML
     */
    extractMainContent(html) {
        // Find the manuscript-content div and extract only its contents (exclude navigation)
        const startMatch = html.match(/<div[^>]*id=["']manuscript-content["'][^>]*>/i);
        if (!startMatch) {
            throw new Error('Could not find manuscript-content div');
        }
        
        const startIndex = startMatch.index + startMatch[0].length;

        // Prefer to stop before the manuscript navigation component, if present
        const remainder = html.substring(startIndex);
        const navMatch = remainder.match(/<nav[^>]*class=["'][^"']*manuscript-nav[^"']*["'][^>]*>/i);

        let endIndex;
        if (navMatch) {
            endIndex = startIndex + navMatch.index;
        } else {
            // Fallback: stop at the closing main tag
            const endMatch = html.match(/<\/main>/i);
            if (!endMatch) {
                throw new Error('Could not find closing main tag');
            }
            endIndex = endMatch.index;
        }
        
        // Extract the content between these points
        const content = html.substring(startIndex, endIndex);
        
        return content;
    }

    /**
     * Convert the built HTML site to markdown
     */
    async convertSiteToMarkdown() {
        console.log('🔄 Converting HTML site to markdown...');
        
        try {
            // Read the built HTML file
            const htmlPath = path.join(__dirname, 'dist', 'index.html');
            if (!fs.existsSync(htmlPath)) {
                throw new Error('Built HTML file not found. Please run "npm run build" first.');
            }
            
            const html = fs.readFileSync(htmlPath, 'utf8');
            
            // Extract metadata
            const metadata = this.extractMetadata(html);
            
            // Extract main content
            const mainContent = this.extractMainContent(html);
            
            // Convert to markdown
            const markdownContent = this.htmlToMarkdown(mainContent);
            
            // Build the complete markdown document
            const markdown = this.buildMarkdownDocument(metadata, markdownContent);
            
            // Write to file — use version from metadata for filename
            const versionSlug = metadata.version
                .replace(/\s*\([^)]*\)/g, '')  // strip "(Valencia)"
                .trim()
                .replace(/\s+/g, '-');
            const outputName = `37-TEP-ARP-${versionSlug}-Pasadena.md`;
            const outputPath = path.join(__dirname, '..', outputName);
            fs.writeFileSync(outputPath, markdown, 'utf8');

            // Keep the shared collection archive (../manuscripts/) in sync
            try {
                const sharedArchiveDir = path.join(__dirname, '..', '..', 'manuscripts');
                if (fs.existsSync(sharedArchiveDir)) {
                    const archiveName = path.basename(outputPath);
                    const paperPrefix = archiveName.split('-')[0];
                    for (const staleFile of fs.readdirSync(sharedArchiveDir)) {
                        if (staleFile !== archiveName && staleFile.endsWith('.md') && staleFile.startsWith(`${paperPrefix}-TEP`)) {
                            fs.rmSync(path.join(sharedArchiveDir, staleFile));
                        }
                    }
                    fs.copyFileSync(outputPath, path.join(sharedArchiveDir, archiveName));
                    console.log(`📄 Copied to shared archive: manuscripts/${archiveName}`);
                }
            } catch (archiveError) {
                console.warn(`⚠️  Could not update shared manuscripts archive: ${archiveError.message}`);
            }
            
            console.log('✅ Markdown conversion complete!');
            console.log(`📄 Output: ${outputPath}`);
            console.log(`📊 Document: ${metadata.title}`);
            console.log(`👤 Author: ${metadata.author}`);
            console.log(`📅 Version: ${metadata.version}`);
            
            return outputPath;
            
        } catch (error) {
            console.error('❌ Markdown conversion failed:', error.message);
            process.exit(1);
        }
    }

    /**
     * Build the complete markdown document with metadata
     */
    buildMarkdownDocument(metadata, content) {
        const now = new Date();
        const timestamp = now.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
        
        // Clean up the title to remove the author part
        const cleanTitle = metadata.title.replace(' | Matthew Lukin Smawfield', '');
        
        // Format date: keep original publish date, add update date
        let formattedDate = metadata.date;
        if (formattedDate.includes('First published:')) {
            formattedDate = formattedDate.replace('First published: ', '');
        }
        // Extract just the original date part before any separator
        const dateParts = formattedDate.split(' · ');
        const originalDate = dateParts[0].trim();
        
        return `# ${cleanTitle}
**${metadata.author}**
Version: ${metadata.version}
First published: ${originalDate} · Last updated: ${timestamp}
DOI: ${metadata.doi}

---

${content}

---

*This document was automatically generated from the TEP-ARP research site. For the interactive version with figures and enhanced formatting, visit: https://mlsmawfield.com/tep/arp/*

*Related Work:*
- [TEP Theory](https://doi.org/10.5281/zenodo.16921911) (Foundational framework)
- [TEP-GNSS: Distance-Structured Correlations in GNSS Clocks](https://doi.org/10.5281/zenodo.17127229) (Paper 1 — GNSS clock-network test of synchronization holonomy)
- [TEP-GNSS-II: 25-Year Analysis of CODE Precise Clock Products](https://doi.org/10.5281/zenodo.17517141) (Paper 2 — quarter-century GNSS clock correlations)
- [TEP-GNSS-RINEX: Raw RINEX Consistency Test](https://doi.org/10.5281/zenodo.17860166) (Paper 3 — raw RINEX-level consistency validation)
- [TEP-H0: The Cepheid Bias](https://doi.org/10.5281/zenodo.18209702) (Companion paper — quantitative Cepheid bias analysis)
- [TEP-EXP: Precision Tests of GR](https://doi.org/10.5281/zenodo.18109760) (Measurement taxonomy)
- [TEP-JWST: JWST High-Redshift Anomalies](https://doi.org/10.5281/zenodo.19000827) (High-redshift galaxy ages and masses)
- [TEP-HUB: The Mount Wilson Paradigm](https://doi.org/10.5281/zenodo.21954258) (Cosmological redshift as temporal shear)
- [TEP-BBN: Primordial Deuterium](https://doi.org/10.5281/zenodo.21841147) (Deuterium isotope identifiability)

*Source code available at: https://github.com/matthewsmawfield/TEP-ARP*
`;
    }
}

// Main execution
async function main() {
    const converter = new HTMLToMarkdownConverter();
    await converter.convertSiteToMarkdown();
}

// Run if called directly
if (require.main === module) {
    main();
}

module.exports = { HTMLToMarkdownConverter };
