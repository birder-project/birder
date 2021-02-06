import { shallowMount } from '@vue/test-utils';
import ImageUpload from '@/components/ImageUpload.vue';

describe('ImageUpload.vue', () => {
  it('ensures component is mountable', () => {
    const wrapper = shallowMount(ImageUpload, {
      propsData: {
        id: 'image-upload',
      },
    });

    expect(wrapper.text()).toContain('Choose a file');
  });
});
