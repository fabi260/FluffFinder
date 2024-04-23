from FluffFinder import vanilla_fluff

text = 'Filmmaking tools to support your creative vision. Unleash your creative instincts, no matter your project or budget. Our cameras and lenses give you the power to shoot a documentary, experimental short film or an entire feature. Produce captivating work with full-frame sensors, built-in filters, professional codecs and more.'

score = vanilla_fluff(text)

print(score)