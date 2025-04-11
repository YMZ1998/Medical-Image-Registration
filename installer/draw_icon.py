import matplotlib.pyplot as plt
from PIL import Image

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axis('off')

ax.text(0, 0.5, 'MIR', fontsize=120, fontweight='bold', va='center', ha='center', alpha=0.8, color='blue')
ax.text(0, -0.5, 'LUNG', fontsize=120, fontweight='bold', va='center', ha='center', alpha=0.8, color='blue')

plt.savefig('mir_lung_logo.png', transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

png_image = Image.open('mir_lung_logo.png')

high_res_sizes = [(256, 256)]
png_image.save('mir_lung_logo.ico', format='ICO', sizes=high_res_sizes)
