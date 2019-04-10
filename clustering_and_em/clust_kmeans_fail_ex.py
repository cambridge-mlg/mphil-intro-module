plt.figure(figsize = (10, 6))
for i in range(4):
    
    picture = plt.imread('ex_{}.jpg'.format(i + 1))
    flat_pic = picture.reshape((-1, 3))
    
    for j, K in enumerate([2, 4, 6]):
        plt.subplot(4, 4, 4*i + j + 1)

        s_pic, mu_pic, losses_pic = k_means(flat_pic, K, 10, np.random.rand(K, 1)*256)
        pic_colors = [np.mean(flat_pic[np.where(np.argmax(s_pic, axis = -1) == k)], axis = 0) for k in np.arange(K)]
        K_colored_pic = flat_pic.copy()

        for k in range(K):
            K_colored_pic[np.where(np.argmax(s_pic, axis = -1) == k)[0], :] = pic_colors[k]

        K_colored_pic = K_colored_pic.reshape(picture.shape)
        plt.imshow(K_colored_pic, origin = 'upper')
        plt.text(720, 480, s = 'K = {} '.format(K), color = 'white', fontsize = 16,
                 horizontalalignment = 'right', verticalalignment = 'bottom')
        remove_axes()

    plt.subplot(4, 4, 4*(i+1))
    plt.imshow(picture, origin = 'upper')
    plt.text(720, 480, s = 'True ', color = 'white', fontsize = 16,
             horizontalalignment = 'right', verticalalignment = 'bottom')
    remove_axes()

plt.tight_layout(w_pad = -3, h_pad = 0)
plt.savefig('clust_kmeans_fail_ex.svg')
plt.show()
